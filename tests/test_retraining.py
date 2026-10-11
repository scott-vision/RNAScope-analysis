import io
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import tifffile

from rnascope_pipeline.cpsam_finetune_gui import (
    CpsamFineTuneApp, PatchItem, _CancelableTrainingHandler, _TrainingCanceled,
)
from rnascope_pipeline.dataset_store import DatasetStore


class RetrainingTests(unittest.TestCase):
    def test_best_checkpoint_selected_instead_of_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / 'model'
            final.touch()
            best = root / 'model_epoch_0020'
            best.touch()
            log = root / 'train.log'
            log.write_text(f'saving model to {final}\n20, train_loss=0.2, test_loss=0.19\n', encoding='utf-8')
            app = CpsamFineTuneApp.__new__(CpsamFineTuneApp)
            app._log = Mock()
            app.trained_model_path_var = Mock()
            app.root = SimpleNamespace(after=lambda delay, callback: callback())
            result = app._postprocess_training_output(log, str(final), best_val=True)
            self.assertEqual(result, str(best.resolve()))
            app.trained_model_path_var.set.assert_called_once_with(result)
            self.assertEqual(app._postprocess_training_output(log, str(final)), str(final.resolve()))

    def test_partial_manual_splits_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DatasetStore(root)
            image, mask = root / 'image.tif', root / 'mask.tif'
            tifffile.imwrite(image, np.ones((8, 8), dtype=np.uint16))
            tifffile.imwrite(mask, np.ones((8, 8), dtype=np.uint16))
            ids = {}
            for split in ('train', 'val', 'test', 'unassigned'):
                ids[split] = store.add_patch(
                    patch_img_path=image, patch_mask_path=mask, source_id='source',
                    source_image_path=image, source_image_hash='hash',
                    x=0, y=0, w=8, h=8, seed=1, split=split,
                )
            snap = store.build_snapshot(name='test', source_ids=None,
                approved_only=False, include_only=True, split_ratios=(0.8, 0.1, 0.1),
                stratify_by_source=True, seed=42)
            for split in ('train', 'val', 'test'):
                self.assertTrue((store.snapshots_dir / snap / split / f'{ids[split]}_img.tif').is_file())

    def test_autolabel_preserves_corrected_masks(self):
        app = CpsamFineTuneApp.__new__(CpsamFineTuneApp)
        items = [PatchItem(status, Path('image'), Path(status), annotation_status=status)
                 for status in ('auto', 'edited', 'approved', 'rejected')]
        app._load_patch_pairs = lambda: items
        app._set_patch_progress_threadsafe = Mock()
        app._log = Mock()
        app.dapi_index_var = SimpleNamespace(get=lambda: '0')
        model = SimpleNamespace(eval=lambda img: (np.ones((8, 8)), None, None))
        app._load_cpsam_model = lambda: model
        app.root = SimpleNamespace(after=lambda *args: None)
        app._run_background = lambda job: job()
        with patch('rnascope_pipeline.cpsam_finetune_gui.tiff.imread', return_value=np.ones((8, 8))), \
                patch('rnascope_pipeline.cpsam_finetune_gui.tiff.imwrite') as write:
            app.autolabel_train_patches()
        self.assertEqual(write.call_count, 1)
        self.assertEqual(write.call_args.args[0], 'auto')

    def test_busy_training_does_not_create_run(self):
        app = CpsamFineTuneApp.__new__(CpsamFineTuneApp)
        app.worker_thread = SimpleNamespace(is_alive=lambda: True)
        app._get_store = Mock()
        with patch('rnascope_pipeline.cpsam_finetune_gui.messagebox.showwarning'):
            app.run_training_command()
        app._get_store.assert_not_called()

    def test_cancel_interrupts_training_log_boundary(self):
        handler = _CancelableTrainingHandler(io.StringIO(), lambda: True)
        record = logging.LogRecord('cellpose.train', logging.INFO, '', 0, 'epoch progress', (), None)
        with self.assertRaises(_TrainingCanceled):
            handler.emit(record)

    def test_close_is_blocked_during_background_task(self):
        app = CpsamFineTuneApp.__new__(CpsamFineTuneApp)
        app.worker_thread = SimpleNamespace(is_alive=lambda: True)
        app.root = Mock()
        with patch('rnascope_pipeline.cpsam_finetune_gui.messagebox.showwarning'):
            app._on_close()
        app.root.destroy.assert_not_called()

    def test_api_cancel_saves_partial_checkpoint(self):
        from cellpose import models, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = CpsamFineTuneApp.__new__(CpsamFineTuneApp)
            app.training_cancel_requested = False
            app._log = Mock()
            app._load_labeled_pairs = lambda folder: ([np.ones((8, 8))], [np.ones((8, 8), dtype=np.int32)])
            net = SimpleNamespace(save_model=lambda path: Path(path).touch())
            def fake_train(net, train_data=None, train_labels=None):
                app.training_cancel_requested = True
                logging.getLogger('cellpose.train').info('20, train_loss=0.2, test_loss=0.19')
                self.fail('Cancellation should interrupt the API call')
            cfg = dict(train_root=root, n_test=0, gpu=False, learning_rate=1e-5,
                weight_decay=0.1, n_epochs=100, train_batch_size=1, min_train_masks=1,
                model_name_out='model', save_each=True, run_id='test')
            logger = logging.getLogger()
            old_handlers = list(logger.handlers)
            with patch.object(models, 'CellposeModel', return_value=SimpleNamespace(net=net)), \
                    patch.object(train, 'train_seg', fake_train):
                with self.assertRaises(_TrainingCanceled):
                    app._run_cellpose_training_api(cfg, io.StringIO())
            self.assertTrue((root / 'models' / 'model_canceled_test').is_file())
            self.assertEqual(logger.handlers, old_handlers)


if __name__ == '__main__':
    unittest.main()
