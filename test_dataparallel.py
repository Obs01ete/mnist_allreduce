import unittest

import torch_testing as tt

from train_model import Trainer, parse_args


class TestDataParallel(unittest.TestCase):

    def test_dataparallel(self):
        args = parse_args(external_args=[])
        trainer = Trainer(args)

        trainer.reference_model.train(False)
        trainer.dataparallel_model.train(False)

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

                rtol = 2e-2
                atol = 1e-7
                tt.assert_allclose(ref_np[1].data, dp_np[1].data, rtol=rtol, atol=atol)
                if ref_np[1].grad is not None and dp_np[1].grad is not None:
                    tt.assert_allclose(ref_np[1].grad, dp_np[1].grad, rtol=rtol)

        def _check_dp_models_equal():
            dp_model = trainer.dataparallel_model
            for i_model, model in enumerate(dp_model.models):
                if i_model == dp_model.master_model_idx:
                    continue
                master_model_params = dp_model.models[dp_model.master_model_idx].parameters()
                model_params = model.parameters()
                for i_layer, (master_param, secondary_param) in enumerate(zip(master_model_params, model_params)):
                    if i_layer == 0:
                        print(f"Master model and model {i_model}")
                        print(master_param[0, 0, ...])
                        print(secondary_param[0, 0, ...])
                    # Important that after all-reduced gradients are applied,
                    # all replica weights are bit-exactly equal even as float32 values!
                    tt.assert_equal(master_param, secondary_param)

        print("Before step")
        _compare_models()
        _check_dp_models_equal()

        for batch_idx, (data, target) in enumerate(trainer.train_loader):
            data, target = data.to(trainer.device), target.to(trainer.device)

            step_info_ref = trainer.reference_model.step(data, target)
            ref_loss = step_info_ref["loss"]

            step_info_dp = trainer.dataparallel_model.step(data, target)
            dp_loss = step_info_dp["loss"]

            print("After step")
            print(f"Loss, reference={ref_loss} dp={dp_loss}")
            _compare_models()
            _check_dp_models_equal()

            break

        return


if __name__ == "__main__":
    unittest.main()
