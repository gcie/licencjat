
class History(dict):
    def __init__(self):
        self.err_rate = []
        self.primal_loss = []
        self.loss = []
        self.test_err_rate = []
        self.dual = []
        self.predictions_test = []
        self.predictions_data = []
        self.predictions_sequences = []
        self.test_primal_loss = []
        self.test_loss = []
        self.ngram_data_stats = []
        self.ngram_test_stats = []
        self.epochs_done = 0
