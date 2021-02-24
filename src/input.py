import pandas

class Input:
    def description_parser(self, column):
        pass

    def classification_parser(self, column):
        res = []
        for cell in column:
            aux = []
            for text in cell.split('|'):
                aux += text.rltrim()

        return res


    def __init__(self, input_file):
        df = pandas.read_excel(input_file)
        

