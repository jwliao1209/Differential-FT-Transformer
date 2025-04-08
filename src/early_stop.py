
class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        verbose: bool = False,
    ) -> None:

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float, mode: str) -> None:
        score = -current_score if mode == 'min' else current_score

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping triggered.')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
