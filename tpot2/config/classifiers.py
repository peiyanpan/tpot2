from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal


#TODO Conditional search space to prevent invalid combinations of hyperparameters
def get_LogisticRegression_ConfigurationSpace(random_state=None):
 
    space = {
        'solver': Categorical('solver', ['saga','liblinear']), 
        'penalty': Categorical("penalty", ['elasticnet','l1', 'l2']), #TODO workaround to support None option?
        'dual': Categorical("dual", [True, False]),
        'C': Float("C", bounds=(1e-4, 1e4), log=True),
        
        #TODO workaround for including None as a value for class_weight
        'class_weight': Categorical("class_weight", ['balanced']),
        'n_jobs': 1,
        'max_iter': 1000,
    }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_KNeighborsClassifier_ConfigurationSpace(n_samples=10):
        return ConfigurationSpace(

                space = {

                    'n_neighbors': Integer("n_neighbors", bounds=(1, min(100,n_samples)), log=True),
                    'weights': Categorical("weights", ['uniform', 'distance']),
                    'p': Integer("p", bounds=(1, 3)),
                    'metric': Categorical("metric", ['euclidean', 'minkowski']),
                    'n_jobs': 1,
                }
            ) 


def get_DecisionTreeClassifier_ConfigurationSpace(random_state=None, n_featues=20):

    space = {
        'criterion': Categorical("criterion", ['gini', 'entropy']),
        'max_depth': Integer("max_depth", bounds=(1, 2*n_featues)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
        'max_features': Categorical("max_features", [1.0, 'sqrt', 'log2']),
        'min_weight_fraction_leaf': 0.0,
    }
    

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_SVC_ConfigurationSpace(random_state=None):

    space = {
        'kernel': Categorical("kernel", ['poly', 'rbf', 'linear', 'sigmoid']),
        'C': Float("C", bounds=(1e-4, 25), log=True),
        'degree': Integer("degree", bounds=(1, 4)),

        #'class_weight': Categorical("class_weight", [None, 'balanced']), #TODO add class_weight. configspace doesn't allow None as a value.
        'max_iter': 3000,
        'tol': 0.001,
        'probability': Categorical("probability", [True]), # configspace doesn't allow bools as a default value? but does allow them as a value inside a Categorical
    }


    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

#TODO Conditional search spaces
def get_LinearSVC_ConfigurationSpace(random_state=None,):
    space = {
            'penalty': Categorical("penalty", ['l1', 'l2']),
            'loss': Categorical("loss", ['hinge', 'squared_hinge']),
            'dual': Categorical("dual", [True, False]),
            'C': Float("C", bounds=(1e-4, 25), log=True),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )




def get_RandomForestClassifier_ConfigurationSpace(random_state=None):
    space = {
            'n_estimators': 100,
            'criterion': Categorical("criterion", ['gini', 'entropy']),
            'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
            'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
            'bootstrap': Categorical("bootstrap", [True, False]),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_GradientBoostingClassifier_ConfigurationSpace(random_state=None, n_classes=None):
    
    if n_classes is not None and n_classes > 2:
        loss = 'log_loss'
    else:
        loss = Categorical("loss", ['log_loss', 'exponential'])

    space = {
        'n_estimators': 100,
        'loss': loss,
        'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
        'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 200)),
        'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
        'subsample': Float("subsample", bounds=(0.1, 1.0)),
        'max_features': Float("max_features", bounds=(0.1, 1.0)),
        'max_depth': Integer("max_depth", bounds=(1, 10)),

        #TODO include max leaf nodes?
        #TODO validation fraction + n_iter_no_change? maybe as conditional

        'tol': 1e-4,
    }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )


def get_XGBClassifier_ConfigurationSpace(random_state=None,):
    
    space = {
            'n_estimators': 100,
            'learning_rate': Float("learning_rate", bounds=(1e-3, 1), log=True),
            'subsample': Float("subsample", bounds=(0.1, 1.0)),
            'min_child_weight': Integer("min_child_weight", bounds=(1, 21)),
            'max_depth': Integer("max_depth", bounds=(1, 11)),
            'n_jobs': 1,
        }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

def get_LGBMClassifier_ConfigurationSpace(random_state=None,):

    space = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': Categorical("boosting_type", ['gbdt', 'dart', 'goss']),
            'num_leaves': Integer("num_leaves", bounds=(2, 256)),
            'max_depth': Integer("max_depth", bounds=(1, 10)),
            'n_estimators': Integer("n_estimators", bounds=(10, 100)),
            'n_jobs': 1,
        }

    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space=space
    )


def get_ExtraTreesClassifier_ConfigurationSpace(random_state=None):
    space = {
            'n_estimators': 100,
            'criterion': Categorical("criterion", ["gini", "entropy"]),
            'max_features': Float("max_features", bounds=(0.05, 1.00)),
            'min_samples_split': Integer("min_samples_split", bounds=(2, 20)),
            'min_samples_leaf': Integer("min_samples_leaf", bounds=(1, 20)),
            'bootstrap': Categorical("bootstrap", [True, False]),
            'n_jobs': 1,
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )



def get_SGDClassifier_ConfigurationSpace(random_state=None):
    
    space = {
            'loss': Categorical("loss", ['log_loss', 'modified_huber']),
            'penalty': 'elasticnet',
            'alpha': Float("alpha", bounds=(1e-5, 0.01), log=True),
            'learning_rate': Categorical("learning_rate", ['invscaling', 'constant']),
            'l1_ratio': Float("l1_ratio", bounds=(0.0, 1.0)),
            'eta0': Float("eta0", bounds=(0.01, 1.0)),
            'power_t': Float("power_t", bounds=(1e-5, 100.0), log=True),
            'n_jobs': 1,
            'fit_intercept': Categorical("fit_intercept", [True]),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state

    return ConfigurationSpace(
        space = space
    )



def get_MLPClassifier_ConfigurationSpace(random_state=None):
    space = {
            'alpha': Float("alpha", bounds=(1e-4, 1e-1), log=True),
            'learning_rate_init': Float("learning_rate_init", bounds=(1e-3, 1.), log=True),
    }
            
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )

GaussianNB_ConfigurationSpace = {}

def get_BernoulliNB_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'alpha': Float("alpha", bounds=(1e-2, 100), log=True),
            'fit_prior': Categorical("fit_prior", [True, False]),
        }
    )


def get_MultinomialNB_ConfigurationSpace():
    return ConfigurationSpace(
        space = {
            'alpha': Float("alpha", bounds=(1e-3, 100), log=True),
            'fit_prior': Categorical("fit_prior", [True, False]),
        }
    )



def get_AdaBoostClassifier_ConfigurationSpace(random_state=None):
    space = {
            'n_estimators': Integer("n_estimators", bounds=(50, 500)),
            'learning_rate': Float("learning_rate", bounds=(0.01, 2), log=True),
            'algorithm': Categorical("algorithm", ['SAMME', 'SAMME.R']),
            'max_depth': Integer("max_depth", bounds=(1, 10)),
        }
    
    if random_state is not None: #This is required because configspace doesn't allow None as a value
        space['random_state'] = random_state
    
    return ConfigurationSpace(
        space = space
    )