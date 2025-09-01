# Step-by-step debugging to find where it hangs
target remote localhost:2331

# Reset and load
monitor reset 1
load
monitor reset 0

echo \n🔍 STEP-BY-STEP DEBUG TO FIND HANG\n

# Break on main
break main
command 1
    echo \n📍 MAIN - stepping through...\n
    # Don't continue automatically, let's step
end

# Break on first tflm_create_interpreter
break *0x5aac
command 2
    echo \n🧠 INTERPRETER CREATION #%d\n
    echo "  Stepping INTO this function to see where it hangs...\n"
    # Step into function instead of continuing
end

# Start and wait for user commands
echo \nReady to step through execution.\n
echo "Commands:\n"
echo "  c = continue\n" 
echo "  s = step into\n"
echo "  n = next\n"
echo "  finish = finish current function\n"
echo "  bt = backtrace\n\n"

continue