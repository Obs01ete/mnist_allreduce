import unittest

import torch_testing as tt

from train_model import Trainer, parse_args


class TestDataParallel(unittest.TestCase):

    def test_dataparallel(self):
        args = parse_args(external_args=[])
        trainer = Trainer(args)

        def _compare_models():
            for i_layer, (ref_np, dp_np) in enumerate(zip(
                    trainer.reference_model.named_parameters(),
                    trainer.dataparallel_model.named_parameters())):

                if i_layer == 0:
                    print(ref_np[0], dp_np[0])
                    print("Weights:")
                    print(ref_np[1].data[0, 0, ...])
                    print(dp_np[1].data[0, 0, ...])
                    print("Grads:")
                    if ref_np[1].grad is not None:
                        print(ref_np[1].grad[0, 0, ...])
                    else:
                        print("None")
                    if dp_np[1].grad is not None:
                        print(dp_np[1].grad[0, 0, ...])
                    else:
                        print("None")
                    print("")

                rtol = 1e-3
                tt.assert_allclose(ref_np[1].data, dp_np[1].data, rtol=rtol)
                if ref_np[1].grad is not None and dp_np[1].grad is not None:
                    tt.assert_allclose(ref_np[1].grad, dp_np[1].grad, rtol=rtol)

                break

        print("Before step")
        _compare_models()

        for batch_idx, (data, target) in enumerate(trainer.train_loader):
            data, target = data.to(trainer.device), target.to(trainer.device)

            dry_run = False

            step_info_ref = trainer.reference_model.step(data, target, dry_run=dry_run)
            ref_loss = step_info_ref["loss"]

            step_info_dp = trainer.dataparallel_model.step(data, target, dry_run=dry_run)
            dp_loss = step_info_dp["loss"]

            print("After step")
            print(f"Loss, reference={ref_loss} dp={dp_loss}")

            _compare_models()

            break

        return


if __name__ == "__main__":
    unittest.main()
