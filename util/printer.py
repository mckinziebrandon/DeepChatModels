
# pretty printing 4 me.

def describe_list(name, list, print_first_n=5):
    print("type(", name, ") = ", type(list),
          "\t", name, "[:", print_first_n, "] = ", list[:print_first_n],
          sep="")