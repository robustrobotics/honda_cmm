# It's important to make sure the data we are using is appropriate for Gaussian Processes.
import matplotlib.pyplot as plt

def viz_params(train_set):



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mech-type', required=True, choices=['params', 'img'])
    parser.add_argument('--L', type=int, required=True)
    args = parser.parse_args()
    print(args)

    #  Load dataset.
    raw_results = read_from_file('/home/mnosew/workspace/honda_cmm/data/doors_gpucb_100L_100M_set0.pickle')
    results = [bb[::] for bb in raw_results[:args.L]]  
    results = remove_duplicates(results)
    results = [item for sublist in results for item in sublist][::2]
    data = parse_pickle_file(results)
    
    val_results = [bb[::] for bb in raw_results[81:]]
    val_results = [item for sublist in val_results for item in sublist]
    val_data = parse_pickle_file(val_results)
    val_set = setup_data_loaders(data=val_data, batch_size=1, single_set=True)

    train_set = setup_data_loaders(data=data, batch_size=len(data), single_set=True)
