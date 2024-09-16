import pandas as pd

def retrieve_training_data(train_path):
    # !TODO: refactor
    def reshape_row(row):
        data = {col: [] for col in row.axes[0] if col in
                ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description', 'image_paths']}
        data["level"] = []
        data["condition"] = []
        data["severity"] = []

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description',
                              'image_paths']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)
            else:
                # !TODO: Seriously, refactor
                for i in range(25):
                    data[column].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + 'train.csv')
    train_desc = pd.read_csv(train_path + 'train_series_descriptions.csv')

    train_df = pd.merge(train, train_desc, on="study_id")

    train_df = pd.concat([reshape_row(row) for _, row in train_df.iterrows()], ignore_index=True)
    train_df['severity'] = train_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    return train_df
