from prompt_utils import prompt_path_dict

print("\nAvailable Prompts:\n")
for m in sorted(prompt_path_dict.keys()):
    print("\t{}".format(m))

print("\nTo add more prompt sets, add a new `*.txt` file to `./prompts` with one prompt per line.\n")