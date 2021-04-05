import pandas as pd

from argparse import ArgumentParser

def main (input_path,
         import_NCCR):
    if import_NCCR:
        table_text = pd.read_csv(f'{input_path}\\Data\\NCCR_Content\\NCCR_Content\\Text_Table.txt', delimiter="\t", encoding="ISO-8859-1")

        print(table_text)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input

    main(input_path=input_path,
         import_NCCR=True)



