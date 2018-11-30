import csv


class Csvreader:
    def __init__(self, file):
        self.file = file
        self.read = self.read_file()

    def read_file(self):
        with open(self.file, 'r', encoding='euc-kr', newline='') as f:
            reader = csv.reader(f)
            read = list(reader)
            return read

    def get_data(self, row):
        date = self.read[row][0]
        ranks = []
        for j in range(1, 7):
            ranks.append(self.read[row + j])
        return date, ranks


if __name__ == '__main__':
    test = Csvreader("data.csv")
    test.read_file()