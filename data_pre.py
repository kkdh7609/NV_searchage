import csvreader
import csv

class Preproc:
    def __init__(self, file, new_file):
        self.file = file
        self.reader = csvreader.Csvreader(file)
        f = open(new_file, 'w', newline='')
        self.writer = csv.writer(f)

    def get_data(self):
        for i in range(0, len(self.reader.read), 7):
            date, ranks = self.reader.get_data(i)
            date = date.split(' ')[1].split(':')
            datas = self.set_data(ranks)
            self.save_file(date, datas)

    def set_data(self, ranks):
        datas = []
        for j in range(0, 10):
            sets = []
            for i in range(1, 6):
                try:
                    sets.append(ranks[i].index(ranks[0][j]) + 1)
                except ValueError:
                    sets.append(-1)
            sets.append(j+1)
            datas.append(sets)
        return datas

    def save_file(self, date, datas):
        data_set = []
        for i in range(0, len(datas)):
            data_set.append(date + datas[i])

        for data in data_set:
            self.writer.writerow(data)

    def __exit__(self):
        self.f.close()


if __name__ == '__main__':
    pre = Preproc("data.csv", 'train_set.csv')
    pre.get_data()
