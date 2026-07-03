import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 1. Load Datasets
# As per instructions, read from the exact dataset path
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Preserve identifiers for submission
test_user_id = test_df['user_id'].copy()
test_video_id = test_df['video_id'].copy()

# Define Target and Drop Identifiers from features
target_col = 'target'
drop_cols = ['user_id', 'video_id', target_col]
features = [c for c in train_df.columns if c not in drop_cols]

# 2. Categorical Encoding
categorical_cols = [
    'workout_type', 
    'preferred_workout_time', 
    'time_of_day', 
    'subscription_type'
]

# Combine briefly to fit label encoders consistently across train and test
for col in categorical_cols:
    if col in features:
        le = LabelEncoder()
        # Fill na just in case to prevent encoding errors
        train_df[col] = train_df[col].astype(str).fillna('missing')
        test_df[col] = test_df[col].astype(str).fillna('missing')
        
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

X = train_df[features]
y = train_df[target_col]
X_test = test_df[features]

# 3. Model Training with Cross-Validation
# Stratified K-Fold setup to reliably validate AUC-ROC
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'random_state': 42
}

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict on test data (accumulate probability scores)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

# 4. Create and Save Submission File
# Standard formatting matching required sample_submission layout (3000 x 3)
submission = pd.DataFrame({
    'user_id': test_user_id,
    'video_id': test_video_id,
    'target': test_preds
})

# Mandatory conversion to CSV as per rules
submission.to_csv('submission.csv', index=False)
