// Adds an "OfflineMemoryAllocation" metadata entry to a .tflite, leaving every
// weight, operator and tensor untouched. Greedy-by-size placement over the
// same tensor lifetimes TFLM derives (inputs live from scope 0, op i is scope
// i+1, graph outputs live to the end). Usage: add_offline_plan in.tflite out.tflite
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

using namespace tflite;

static int ElemBytes(TensorType t) {
  switch (t) {
    case TensorType_INT8: case TensorType_UINT8: case TensorType_BOOL: return 1;
    case TensorType_INT16: case TensorType_FLOAT16: return 2;
    case TensorType_INT32: case TensorType_FLOAT32: return 4;
    default: return 8;
  }
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream in(argv[1], std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
  ModelT model;
  GetModel(raw.data())->UnPackTo(&model);
  if (model.subgraphs.size() != 1) { std::fprintf(stderr, "expected 1 subgraph\n"); return 3; }
  SubGraphT& sg = *model.subgraphs[0];
  const int n = static_cast<int>(sg.tensors.size());
  const int scopes = static_cast<int>(sg.operators.size());
  std::vector<int> first(n, -1), last(n, -1), size(n, 0);
  std::vector<bool> plan(n, false);
  for (int i = 0; i < n; ++i) {
    const TensorT& t = *sg.tensors[i];
    int64_t count = 1;
    for (int d : t.shape) count *= d;
    size[i] = static_cast<int>((count * ElemBytes(t.type) + 15) / 16 * 16);
    const bool is_const = t.buffer != 0 && !model.buffers[t.buffer]->data.empty();
    plan[i] = !is_const && !t.is_variable && count != 0;
  }
  auto touch = [&](int t, int scope, bool create) {
    if (t < 0) return;
    if (create && (first[t] < 0 || scope < first[t])) first[t] = scope;
    if (last[t] < scope) last[t] = scope;
  };
  for (int t : sg.inputs) { touch(t, 0, true); }
  for (int o = 0; o < scopes; ++o) {
    for (int t : sg.operators[o]->outputs) touch(t, o + 1, true);
    for (int t : sg.operators[o]->inputs) touch(t, o + 1, false);
    for (int t : sg.operators[o]->outputs) touch(t, o + 1, false);
  }
  for (int t : sg.outputs) { touch(t, scopes, true); }

  std::vector<int> order;
  for (int i = 0; i < n; ++i) if (plan[i] && first[i] >= 0) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return size[a] > size[b]; });
  std::vector<int32_t> offset(n, -1);
  std::vector<int> placed;
  int32_t high = 0;
  for (int t : order) {
    int32_t best = 0;
    // First-fit over placed buffers that overlap in time, sorted by offset.
    std::vector<int> live;
    for (int p : placed)
      if (!(last[p] < first[t] || last[t] < first[p])) live.push_back(p);
    std::sort(live.begin(), live.end(), [&](int a, int b) { return offset[a] < offset[b]; });
    for (int p : live) {
      if (offset[p] - best >= size[t]) break;
      best = std::max(best, offset[p] + size[p]);
    }
    offset[t] = best;
    placed.push_back(t);
    high = std::max(high, best + size[t]);
  }
  std::fprintf(stderr, "planned %zu tensors, activation arena %d bytes\n", order.size(), high);

  auto buf = std::make_unique<BufferT>();
  std::vector<int32_t> words = {1, 0, n};
  words.insert(words.end(), offset.begin(), offset.end());
  buf->data.resize(words.size() * 4);
  std::memcpy(buf->data.data(), words.data(), buf->data.size());
  model.buffers.push_back(std::move(buf));
  auto meta = std::make_unique<MetadataT>();
  meta->name = "OfflineMemoryAllocation";
  meta->buffer = static_cast<uint32_t>(model.buffers.size() - 1);
  model.metadata.push_back(std::move(meta));

  flatbuffers::DefaultAllocator allocator;
  flatbuffers::FlatBufferBuilder fbb(1 << 20, &allocator);
  FinishModelBuffer(fbb, Model::Pack(fbb, &model));
  std::ofstream out(argv[2], std::ios::binary);
  out.write(reinterpret_cast<const char*>(fbb.GetBufferPointer()), fbb.GetSize());
  return out ? 0 : 4;
}
