from utils.utils import extract_dataset
from common.eval import *


model.eval()

if P.mode == 'test_clean_acc':
    from evals import test_classifier
    test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_adv_acc':
    from evals import test_classifier_adv

    test_classifier_adv(P, model, test_loader, 0,
                        adversary=adversary_t, logger=None, ret='adv')

elif P.mode == 'test_auto_attack':
    from autoattack import AutoAttack

    auto_adversary = AutoAttack(model, norm=P.distance, eps=P.epsilon, version='standard')
    x_test, y_test = extract_dataset(test_loader)

    x_adv = auto_adversary.run_standard_evaluation(x_test, y_test)

elif P.mode == 'test_mce':
    from evals import test_classifier

    mean_corruption_error = 0.
    for name in corruption_list:
        error = test_classifier(P, model, corruption_loader[name], 0, logger=None)
        mean_corruption_error += error
        print (f'Error of {name}: {error}%\n')
    print (f'MCE: {mean_corruption_error/len(corruption_list)} %')

else:
    raise NotImplementedError()
