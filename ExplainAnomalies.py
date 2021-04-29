import argparse
from ExplainCircuitRecords import *
from AnomalyDetector import AnomalyDetector
from SimpleAnomalyDetector import SimpleAnomalyDetector as SimpleAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_noise', default=0, type=int)
    parser.add_argument('--prec_back', default=1.0, type=float)
    parser.add_argument('--circuit', default='', type=str)
    parser.add_argument('--anomaly', default='', type=str)
    args = parser.parse_args()
    circuit = args.circuit
    modified_circuit = args.anomaly
    n_noise = args.n_noise
    background_proportion = args.prec_back

    if modified_circuit == '' or circuit == '':
        raise ValueError('Circuit and anomalous circuit must be selected')

    background_selection = 'Background- {}%'.format(background_proportion*100)
    noise = '{} noise features'.format(n_noise)

    # initialize the anomaly detector
    simple_ad_model = SimpleAD()
    anomaly_detector = AnomalyDetector(simple_ad_model)

    exp_eval = ExplainCircuitRecords(anomaly_detector, amount_of_noise=n_noise, background_proportion=background_proportion)

    # set the circuit and read the data truth tables
    exp_eval.set_circuit(circuit)
    modified_data = exp_eval.read_data(modified_circuit)

    # train the model
    exp_eval.fit(modified_data)

    # set the chosen explanation method
    explainer = 'KernelSHAP'
    exp_eval.set_explainer(explainer)

    # set a list with chosen evaluation metrics
    metrics = ['RPrecision', 'MRR', 'MAP']
    exp_eval.set_metrics(metrics)

    method_final_scores = exp_eval.explain_unsupervised_data(modified_circuit)

    full_path_metrics_method = os.path.join(METHOD_DICT[explainer] + background_selection + '/' + noise + '/'
                                            + circuit + '/' + 'metrics' + '.csv')
    # save the metrics results
    metrics_df = pd.DataFrame({
        'circuit name': [circuit],
        'anomaly': [modified_circuit]
    })
    for metric in metrics:
        metrics_df[metric] = method_final_scores[metric]

    if os.path.exists(full_path_metrics_method):
        metrics_df.to_csv(full_path_metrics_method, mode='a', index=False, header=False)
    else:
        metrics_df.to_csv(full_path_metrics_method, index=False)


if __name__ == "__main__":
    main()

print('done')




