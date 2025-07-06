class EarlyStopping:
    """
    mode: 'min'(lower is better) -> Val_Loss, 'max' (higher is better) -> Accuracy.
    """
    def __init__(self, patience: int = 10, delta: float = 0.0001, mode: str = 'min') -> None:

        # Check the mode of Early Stopping else raise Error
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0

        # Initialize best_score based on the mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, score: float) -> bool:
        improved = (
            (self.mode == 'min' and score < self.best_score - self.delta) or
            (self.mode == 'max' and score > self.best_score + self.delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience