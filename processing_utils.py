from MLP.MLP import MLP


def start_test_input(dataset: MLP):
    while True:
        inp = input(f"\nType index for test from 0 to {dataset.get_test_length() - 1}\n")
        if inp == "":
            break
        try:
            index = int(inp)
            max_index = dataset.get_test_length() - 1
            if index < 0 or index > max_index:
                print(f"[start_test_input] wrong index {index}. Index must be from 0 to {max_index}")
                continue
            dataset.predict(int(inp))
        except:
            print(f"Wrong value {inp}")
