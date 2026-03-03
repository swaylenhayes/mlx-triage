# VHS demo recording for mlx-triage README
#
# Setup (run once before recording):
#   cp -r ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots/08231374eeacb049a0eade7922910865b8fce912 /tmp/Llama-3.2-1B-Instruct-4bit
#
# Record:
#   vhs docs/assets/demo.tape

Output docs/assets/demo.gif

Set FontSize 14
Set Width 1000
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 20

# Show the command being typed
Type "uv run mlx-triage check /tmp/Llama-3.2-1B-Instruct-4bit"
Sleep 500ms
Enter

# Wait for the output to render
Sleep 3s

# Pause so viewer can read the results
Sleep 4s
