import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct

np.random.seed(1)

class GPLearner:
    #X : n_samples x n_features
    #y : n_samples x 1 (0/1 classification)
    def clean_data(self, datasets):
        # get number of samples
        n_samples = 0
        for dataset in datasets:
            n_samples += len(dataset.keys())
        n_features = 3 # RGB values for color feature
        X = np.zeros((n_samples, n_features))
        y = np.zeros(n_samples)
        sample = 0
        for dataset in datasets:
            for handle_name in dataset.keys():
                X[sample,:] = dataset[handle_name][0].color
                joint_type = dataset[handle_name][1].joint_type
                if joint_type == 'prismatic':
                    y[sample] = 0.
                elif joint_type == 'revolute':
                    y[sample] = 1.
                sample += 1
        return X, y

    # for now input is 3D (RGB) and output is 2D (pris,rev)
    def get_model(self, datasets):
        gp = GaussianProcessClassifier(kernel=RBF(10, (1e-2, 1e2)), n_restarts_optimizer=9)
        # Fit to data using Maximum Likelihood Estimation of the parameters

        X, y = self.clean_data(datasets)
        gp.fit(X, y)

        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        n = 100
        x0_min = x1_min = -.1
        x0_max = x1_max = 1.1
        x0 = x1 = np.linspace(-.1, 1.1, n)
        xx, yy = np.meshgrid(x0, x1)
        Xfull = np.zeros((n*n, 3))

        # since only using colors R and G, the B input to predict will always by 0 for now
        Xfull[:,:2] = np.c_[xx.ravel(), yy.ravel()]
        Xfull[:,2] = np.zeros(n*n)

        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred_prob = gp.predict_proba(Xfull)
        y_pred = gp.predict(Xfull)

        # Plot the probability of revolute given an input RG(B=0) value
        plt.figure()
        plt.scatter(X[:,0], X[:,1], c=X, marker='x', linewidths=.00001)
        im_handle = plt.imshow(y_pred_prob[:,1].reshape((n,n)), extent=(x0_min, x0_max, x1_min, x1_max), origin='lower')
        plt.xlabel('R value (of RGB)')
        plt.ylabel('G value (of RGB)')
        plt.title('Probabilit of Revolute Output Type based on RG(B=0) Values')
        plt.colorbar(im_handle)

        # Plot the predicted output value of a given RG(B=0) input
        plt.figure()
        plt.plot(1,0,'rx', label='prismatic data')
        plt.plot(0,1,'gx', label='revolute data')
        im_handle = plt.imshow(y_pred.reshape((n,n)), extent=(x0_min, x0_max, x1_min, x1_max), origin='lower')
        plt.xlabel('R value (of RGB)')
        plt.ylabel('G value (of RGB)')
        plt.colorbar(im_handle)
        plt.legend()
        plt.title('Joint Classification based on RG(B=0) Values')
        plt.xlim(-.1, 1.1)
        plt.ylim(-.1, 1.1)

        plt.show(block=True)
