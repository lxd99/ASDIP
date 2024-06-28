import pandas as pd
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import pickle as pk

def get_seq_data(data, all_labels, logger, type):
    seq_data = dict()
    if type == 'last':
        train_label = all_labels['train'][-1:]
    elif type == 'all':
        train_label = all_labels['train']
    else:
        raise ValueError("Not iImplemented data split")
    m_seq_data = {'id': [], 'cascade': [], 'time': [], 'label': [], 'predict_time': []}
    for m_train_label in tqdm(train_label, desc='get seq training data'):
        m_seq_data['id'].append(m_train_label['id'])
        m_seq_data['label'].extend(m_train_label['label_user'])
        m_seq_data['predict_time'].extend([pd.to_datetime(m_train_label['predict_time']).
                                          value // 10 ** 9] * len(m_train_label['id']))
        print(m_train_label['predict_time'], len(m_train_label['id']))
        train_data = data[pd.to_datetime(data['time'], unit='s') <
                          pd.to_datetime(m_train_label['predict_time'])]
        data_before_predict_len = len(train_data)
        assert len(set(train_data['cas']) & set(m_train_label['id'])) == len(
            set(m_train_label['id'])), 'some cascades do not appear before observe time'
        cas_dict = dict()
        for cas, df in train_data.groupby('cas'):
            cas_dict[cas] = (list(df['dst']), list(df['time']))
        for cas in m_train_label['id']:
            m_seq_data['cascade'].append(cas_dict[cas][0])
            m_seq_data['time'].append(cas_dict[cas][1])
    m_seq_data['id'] = np.concatenate(m_seq_data['id'], axis=0)
    m_seq_data['predict_time'] = np.array(m_seq_data['predict_time'], dtype=np.float32)
    seq_data['train'] = m_seq_data
    avg_cas_len = np.mean([len(x) for x in m_seq_data['cascade']])
    avg_label_len = np.mean([len(x) for x in m_seq_data['label']])
    logger.info(f'train:')
    logger.info(f"cas num is {len(m_seq_data['id'])}, interaction num is {data_before_predict_len},"
                f"avg cas len is {avg_cas_len}, avg label len is {avg_label_len}")
    for dtype in ['val', 'test']:
        # convert conflict of pad number
        m_seq_data = dict()
        id = all_labels[dtype]['id']
        label = all_labels[dtype]['label_user']
        label_dict = dict(zip(id, label))
        predict_start = pd.to_datetime(all_labels[f'{dtype}_time'])
        data_before_predict = data[pd.to_datetime(data['time'], unit='s') < predict_start]
        data_before_predict_len = len(data_before_predict)
        data_before_predict = data_before_predict[data_before_predict['cas'].isin(set(id))]
        m_ids, m_cascades, m_times, m_labels = [], [], [], []
        for cas, df in tqdm(data_before_predict.groupby(by='cas', as_index=False), desc=f'get {dtype} data'):
            df.sort_values(by='time', inplace=True)
            m_ids.append(cas)
            m_cascades.append(list(df['dst']))
            m_times.append(list(df['time']))
            m_labels.append(label_dict[cas])
        m_seq_data['id'] = np.array(m_ids)
        m_seq_data['cascade'] = m_cascades
        m_seq_data['time'] = m_times
        m_seq_data['label'] = m_labels
        m_seq_data['predict_time'] = np.array([predict_start.value // 10 ** 9] * len(m_ids), dtype=np.float32)
        seq_data[dtype] = m_seq_data
        avg_cas_len = np.mean([len(x) for x in m_cascades])
        avg_label_len = np.mean([len(x) for x in m_labels])
        assert len(set(m_ids)) == len(set(id)), 'There are some cascades that can not be found!'
        logger.info(f'{dtype}:')
        logger.info(f"cas num is {len(m_seq_data['id'])}, interaction num is {data_before_predict_len},"
                    f"avg cas len is {avg_cas_len}, avg label len is {avg_label_len}")
    return seq_data
