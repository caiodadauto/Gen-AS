from tqdm import tqdm


class ProgressBar:
    def __init__(self, total, progress_bar_qt=None, desc=None):
        self.count = 0
        if progress_bar_qt is None:
            self.gui = False
            self.bar = tqdm(total=total, desc=desc)
        else:
            self.gui = True
            self.bar = progress_bar_qt
            self.bar.emit(f"max|{total}")
            self.bar.emit(f"desc|{desc}")

    def update(self, n=1):
        self.count += n
        if self.gui:
            self.bar.emit(f"set|{self.count}")
        else:
            self.bar.update(n)

    def set_postfix(self, **kwargs):
        if not self.gui:
            self.bar.set_postfix(**kwargs)

    def set_sufix(self, desc):
        if self.gui:
            self.bar.emit(f"sufix|{desc}")

    def close(self):
        if not self.gui:
            self.bar.close()
        elif not self.bar.stop_running:
            self.bar.emit("set|0")
            self.bar.emit("sufix|")
