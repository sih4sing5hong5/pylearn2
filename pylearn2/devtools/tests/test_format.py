import os
import pylearn2
from pylearn2.devtools.list_files import list_files

whitelist = ["rbm_tools.py",
             "training_algorithms/tests/test_bgd.py",
             "training_algorithms/tests/test_sgd.py",
             "training_algorithms/tests/test_learning_rule.py",
             "training_algorithms/bgd.py",
             "distributions/mnd.py",
             "models/sparse_autoencoder.py",
             "models/tests/test_dbm.py",
             "models/tests/test_autoencoder.py",
             "models/tests/test_s3c_inference.py",
             "models/tests/test_maxout.py",
             "models/tests/test_mnd.py",
             "models/tests/test_s3c_misc.py",
             "models/gsn.py",
             "models/model.py",
             "models/dbm/layer.py",
             "models/dbm/__init__.py",
             "models/dbm/dbm.py",
             "models/dbm/ising.py",
             "models/dbm/inference_procedure.py",
             "models/differentiable_sparse_coding.py",
             "models/local_coordinate_coding.py",
             "models/maxout.py",
             "models/s3c.py",
             "models/mnd.py",
             "models/dense_binary_dbm.py",
             "models/svm.py",
             "models/rbm.py",
             "models/autoencoder.py",
             "tests/test_monitor.py",
             "tests/rbm/test_ais.py",
             "kmeans.py",
             "packaged_dependencies/theano_linear/conv2d.py",
             "packaged_dependencies/theano_linear/imaging.py",
             "packaged_dependencies/theano_linear/pyramid.py",
             "packaged_dependencies/theano_linear/unshared_conv/"
             "test_gpu_unshared_conv.py",
             "packaged_dependencies/theano_linear/unshared_conv/"
             "test_localdot.py",
             "packaged_dependencies/theano_linear/unshared_conv/localdot.py",
             "packaged_dependencies/theano_linear/unshared_conv/"
             "unshared_conv.py",
             "packaged_dependencies/theano_linear/linear.py",
             "packaged_dependencies/theano_linear/test_spconv.py",
             "packaged_dependencies/theano_linear/test_matrixmul.py",
             "packaged_dependencies/theano_linear/spconv.py",
             "expr/tests/test_coding.py",
             "expr/tests/test_normalize.py",
             "expr/tests/test_stochastic_pool.py",
             "expr/nnet.py",
             "expr/stochastic_pool.py",
             "expr/sampling.py",
             "expr/normalize.py",
             "expr/information_theory.py",
             "expr/basic.py",
             "testing/datasets.py",
             "testing/cost.py",
             "gui/graph_2D.py",
             "gui/patch_viewer.py",
             "sandbox/cuda_convnet/weight_acts.py",
             "sandbox/cuda_convnet/filter_acts.py",
             "sandbox/cuda_convnet/tests/test_filter_acts_strided.py",
             "sandbox/cuda_convnet/tests/test_probabilistic_max_pooling.py",
             "sandbox/cuda_convnet/tests/test_filter_acts.py",
             "sandbox/cuda_convnet/tests/test_weight_acts_strided.py",
             "sandbox/cuda_convnet/tests/test_image_acts_strided.py",
             "sandbox/cuda_convnet/tests/test_img_acts.py",
             "sandbox/cuda_convnet/testsprofile_probabilistic_max_pooling.py",
             "sandbox/cuda_convnet/tests/test_weight_acts.py",
             "sandbox/cuda_convnet/tests/test_stochastic_pool.py",
             "sandbox/cuda_convnet/specialized_bench.py",
             "sandbox/cuda_convnet/response_norm.py",
             "sandbox/cuda_convnet/__init__.py",
             "sandbox/cuda_convnet/img_acts.py",
             "sandbox/cuda_convnet/convnet_compile.py",
             "sandbox/cuda_convnet/base_acts.py",
             "sandbox/cuda_convnet/pthreads.py",
             "sandbox/cuda_convnet/pool.py",
             "sandbox/cuda_convnet/bench.py",
             "sandbox/cuda_convnet/stochastic_pool.py",
             "sandbox/cuda_convnet/probabilistic_max_pooling.py",
             "sandbox/tuple_var.py",
             "sandbox/lisa_rl/bandit/average_agent.py",
             "sandbox/lisa_rl/bandit/classifier_bandit.py",
             "sandbox/lisa_rl/bandit/classifier_agent.py",
             "sandbox/lisa_rl/bandit/plot_reward.py",
             "config/old_config.py",
             "config/yaml_parse.py",
             "space/tests/test_space.py",
             "datasets/utlc.py",
             "datasets/mnistplus.py",
             "datasets/cos_dataset.py",
             "datasets/cifar10.py",
             "datasets/svhn.py",
             "datasets/tests/test_csv_dataset.py",
             "datasets/tests/test_icml07.py",
             "datasets/tests/test_utlc.py",
             "datasets/preprocessing.py",
             "datasets/config.py",
             "datasets/dense_design_matrix.py",
             "datasets/adult.py",
             "datasets/tfd.py",
             "datasets/icml07.py",
             "datasets/filetensor.py",
             "datasets/hepatitis.py",
             "datasets/wiskott.py",
             "datasets/mnist.py",
             "datasets/sparse_dataset.py",
             "datasets/csv_dataset.py",
             "datasets/cifar100.py",
             "datasets/tl_challenge.py",
             "datasets/norb_small.py",
             "datasets/retina.py",
             "datasets/ocr.py",
             "datasets/stl10.py",
             "datasets/vector_spaces_dataset.py",
             "datasets/debug.py",
             "datasets/binarizer.py",
             "utils/utlc.py",
             "utils/tests/test_serial.py",
             "utils/common_strings.py",
             "utils/serial.py",
             "utils/mem.py",
             "train.py",
             "format/tests/test_target_format.py",
             "format/target_format.py",
             "dataset_get/dataset-get.py",
             "dataset_get/helper-scripts/make-archive.py",
             "dataset_get/dataset_resolver.py",
             "pca.py",
             "monitor.py",
             "optimization/batch_gradient_descent.py",
             "optimization/test_batch_gradient_descent.py",
             "optimization/minres.py",
             "costs/ebm_estimation.py",
             "costs/gsn.py",
             "costs/mlp/__init__.py",
             "costs/mlp/dropout.py",
             "costs/mlp/missing_target_cost.py",
             "costs/cost.py",
             "costs/dbm.py",
             "costs/autoencoder.py",
             "linear/conv2d.py",
             "linear/local_c01b.py",
             "linear/matrixmul.py",
             "linear/linear_transform.py",
             "linear/conv2d_c01b.py",
             "energy_functions/rbm_energy.py",
             "scripts/plot_monitor.py",
             "scripts/tests/test_autoencoder.py",
             "scripts/show_examples.py",
             "scripts/summarize_model.py",
             "scripts/lcc_tangents/make_dataset.py",
             "scripts/pkl_inspector.py",
             "scripts/print_monitor.py",
             "scripts/show_binocular_greyscale_examples.py",
             "scripts/jobman/tester.py",
             "scripts/dbm/show_samples.py",
             "scripts/dbm/show_reconstructions.py",
             "scripts/dbm/dbm_metrics.py",
             "scripts/dbm/top_filters.py",
             "scripts/papers/maxout/svhn_preprocessing.py",
             "scripts/papers/jia_huang_wkshp_11/fit_final_model.py",
             "scripts/papers/jia_huang_wkshp_11/evaluate.py",
             "scripts/papers/jia_huang_wkshp_11/extract_features.py",
             "scripts/papers/jia_huang_wkshp_11/assemble.py",
             "scripts/gpu_pkl_to_cpu_pkl.py",
             "scripts/datasets/make_cifar10_whitened.py",
             "scripts/datasets/make_cifar100_patches_8x8.py",
             "scripts/datasets/make_cifar100_patches.py",
             "scripts/datasets/make_cifar10_gcn_whitened.py",
             "scripts/datasets/make_cifar100_whitened.py",
             "scripts/datasets/make_stl10_patches_8x8.py",
             "scripts/datasets/make_cifar100_gcn_whitened.py",
             "scripts/datasets/make_stl10_whitened.py",
             "scripts/datasets/make_stl10_patches.py",
             "scripts/gsn_example.py",
             "scripts/tutorials/tests/test_convnet.py",
             "scripts/tutorials/deep_trainer/run_deep_trainer.py",
             "scripts/tutorials/grbm_smd/make_dataset.py",
             "scripts/tutorials/grbm_smd/test_grbm_smd.py",
             "scripts/icml_2013_wrepl/multimodal/"
             "extract_layer_2_kmeans_features.py",
             "scripts/icml_2013_wrepl/multimodal/make_submission.py",
             "scripts/icml_2013_wrepl/multimodal/lcn.py",
             "scripts/icml_2013_wrepl/multimodal/extract_kmeans_features.py",
             "scripts/icml_2013_wrepl/emotions/emotions_dataset.py",
             "scripts/icml_2013_wrepl/emotions/make_submission.py",
             "scripts/icml_2013_wrepl/black_box/black_box_dataset.py",
             "scripts/icml_2013_wrepl/black_box/make_submission.py",
             "scripts/diff_monitor.py",
             "train_extensions/window_flip.py",
             "corruption.py",
             "sandbox/cuda_convnet/tests/profile_probabilistic_max_pooling.py",
             "training_algorithms/sgd.py",
             "devtools/nan_guard.py",
             "models/mlp.py",
             "sandbox/lisa_rl/bandit/gaussian_bandit.py",
             "config/tests/test_yaml_parse.py",
             "utils/iteration.py",
             "utils/track_version.py",
             "scripts/get_version.py"]


def test_format():
    format_infractions = []
    for path in list_files(".py"):
        rel_path = os.path.relpath(path, pylearn2.__path__[0])
        if rel_path in whitelist:
            continue
        with open(path) as file:
            for i, line in enumerate(file):
                if len(line) > 79:
                    format_infractions.append((path, i + 1))
    if len(format_infractions) > 0:
        msg = "\n".join('File "%s" line %d has more than 79 characters'
              % (fn, line) for fn, line in format_infractions)
        raise AssertionError("Format not respected:\n%s" % msg)