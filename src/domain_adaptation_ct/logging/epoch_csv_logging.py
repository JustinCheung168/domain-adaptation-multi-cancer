import csv

from transformers import TrainerCallback, TrainerState

class CSVLoggingCallback(TrainerCallback):
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.header_written = False
        self.epoch_data = {}

    def on_log(self, args, state: TrainerState, control, logs, **kwargs):
        """Store training info internally, to be recorded after validation metrics are captured."""
        if 'loss' in logs: # Use this condition to check if we are performing training.
            assert 'learning_rate' in logs, "Didn't see learning_rate in logs - please make sure the Trainer captures it."

            self.epoch_data[state.epoch] = {
                'epoch': state.epoch,
                'step': state.global_step,
                'train_loss': logs['loss'],
                'learning_rate': logs['learning_rate'],
                'grad_norm': logs['grad_norm']
            }

    def on_evaluate(self, args, state: TrainerState, control, metrics, **kwargs):
        """Capture validation metrics and write both training & validation results to CSV."""
        # Merge evaluation metrics into the epoch record
        epoch_record = self.epoch_data[state.epoch]
        epoch_record.update(metrics)

        # Write to CSV
        with open(self.csv_path, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=epoch_record.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(epoch_record)

