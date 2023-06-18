import argparse

class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-n', '--network', default='Resnet50' ,type=str, help='network to use')

        self.parser.add_argument('-d', '--data_dir', default='data\\test_1000\\', type=str, help='data directory')
        self.parser.add_argument('-f', '--feature_dir', default='feature', type=str, help='feature directory')
        self.parser.add_argument('-r', '--result_dir', default='result', type=str, help='result directory')

        self.parser.add_argument('-k', '--top_k', default=25, type=int, help='retrieved top k images')
        self.parser.add_argument('-im', '--img_test', default='test_all', type=str, help='image to test in directory')
        self.parser.add_argument('-dn', '--divide_n', default=2, type=int, help='divide into n*n blocks')
        
        self.parser.add_argument('-nin', '--not_indexing', action='store_true', help='do not indexing')
        self.parser.add_argument('-sp', '--spliting', action='store_true', help='split images to make features')
        self.parser.add_argument('-m', '--mutual', action='store_true', help='using mutual acknowledge')
        self.parser.add_argument('-np', '--nocopy', action='store_true', help='do not copy img into result folder')

        self.parser.add_argument('-dr', '--split_drop', default=0.2, type=float, help='drop rate of features after split')

    def parse(self):
        return self.parser.parse_args()