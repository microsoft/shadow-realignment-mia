import matplotlib.pyplot as plt
import numpy as np
import time

class Logger():
    def __init__(self, exp, print_every, save_path):
        self.exp = exp
        self.losses = []
        self.train_accs = []
        self.val_accs = []
        self.print_every = print_every
        self.save_path = save_path
        self.start_time = time.time()
            
    
    def log_loss(self, loss, epoch, it):
        self.losses.append(loss)
        if len(self.losses) % self.print_every == 0:
            avg_losses = np.mean(np.array(self.losses).\
                reshape(-1, self.print_every), axis=1)
            plt.plot(range(1, len(self.losses), self.print_every), avg_losses)
            plt.xlim(1, len(self.losses))
            plt.xlabel('Number of batches')
            plt.ylabel(f'Loss (avg. every {self.print_every}) batches')
            plt.savefig(f'{self.save_path}_loss.png')
            plt.close()
            elapsed_time = time.time() - self.start_time
            print(f'[Exp {self.exp+1}/epoch {epoch}]', 
                f'Avg. loss over batches {it-self.print_every+1}-{it}: {avg_losses[-1]:.2f} ({elapsed_time:.0f} secs)')


    def log_accuracy(self, train_acc, val_acc):
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        x = range(0, len(self.train_accs))
        plt.plot(x, self.train_accs, label='Train')
        plt.plot(x, self.val_accs, label='Validation')
        plt.xlim(0, len(self.train_accs))
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.yticks(np.linspace(0, 1, 11))
        plt.savefig(f'{self.save_path}_accuracy.png')
        plt.close()
