from dataset_utils import DataSet


def start_test_input(dataset: DataSet):
    while True:
        inp = input(f"\nType index for test from 0 to {dataset.get_test_length() - 1}\n")
        if inp == "":
            break
        try:
            dataset.show_image(int(inp))
        except:
            print(f"Wrong value {inp}")
