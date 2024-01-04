import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from generators import generator_name_dict

print("\nAvailable Models:\n")
for m in sorted(generator_name_dict.keys()):
    print("\t{}".format(m))

print("\nTo add more models, create a new class file in the `generators` folder and register it in `generators/__init__.py`\n")