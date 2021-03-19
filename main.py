import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA



# in
# [1] aishell-3
aishell_dir = '/ceph/home/hujk17/npy-EarSpeech-HCSI-Data/dereverb_npy'

# [2] cross samples
cross_samples_path = ['/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-000001-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-000016-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-100001-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-100024-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-200001-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-200005-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-LJ001-0001-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-LJ002-0014-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-p225_001-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-p225_003-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-p315_003-GE2E.npy',
                      '/ceph/home/hujk17/Tuned-GE2E-EarSpeech/samples_Cross/spk-p315_023-GE2E.npy',]
cross_samples_speaker_id = [300,
                            300,
                            300,
                            300,
                            300,
                            300,
                            301,
                            301,
                            302,
                            302,
                            303,
                            303,]


N = 16


# out
png_path = 'pca_cross_speaker_emb_' + str(N) + '.png'




def tsne_plotter(data, label, test_data, test_label, save_png, title):
    n_labels = len(set(label))

    # tsne
    # tsne = TSNE(n_components=2, init='pca', learning_rate=10, perplexity=12, n_iter=1000)
    # transformed_data = tsne.fit_transform(data)

    # LDA
    # lda = LDA(n_components=2)
    # transformed_data = lda.fit_transform(data, label)

    # PCA
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)

    test_transformed_data = pca.transform(test_data)

    plt.figure()
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], 30, c=label[:], cmap=plt.cm.Spectral, alpha=0.5)
    # X -> 标贝
    plt.scatter(test_transformed_data[:6, 0], test_transformed_data[:6, 1], 30, c=test_label[:6], marker='x', cmap=plt.cm.Spectral, alpha=0.5)
    # + -> LJSpeech_sub_1
    plt.scatter(test_transformed_data[6:7, 0], test_transformed_data[6:7, 1], 30, c=test_label[6:7], marker='+', cmap=plt.cm.Spectral, alpha=0.5)
    # ^ -> LJSpeech_sub_2
    plt.scatter(test_transformed_data[7:8, 0], test_transformed_data[7:8, 1], 30, c=test_label[7:8], marker='^', cmap=plt.cm.Spectral, alpha=0.5)
    # 正方形 -> VCTK_1_sub_1
    plt.scatter(test_transformed_data[8:9, 0], test_transformed_data[8:9, 1], 30, c=test_label[8:9], marker='s', cmap=plt.cm.Spectral, alpha=0.5)
    # 八角形 -> VCTK_1_sub_2
    plt.scatter(test_transformed_data[9:10, 0], test_transformed_data[9:10, 1], 30, c=test_label[9:10], marker='D', cmap=plt.cm.Spectral, alpha=0.5)
    # 五边形 -> VCTK_1
    plt.scatter(test_transformed_data[10:12, 0], test_transformed_data[10:12, 1], 30, c=test_label[10:12], marker='p', cmap=plt.cm.Spectral, alpha=0.5)
    plt.title(title)
    plt.savefig(save_png)



def data_call():
    speaker_emb = []
    speaker_label = []
    
    test_speaker_emb = []
    test_speaker_label = []
    


    # Ai-shell-3
    speaker_id = 0
    print(os.listdir(aishell_dir))
    for sub_dir in os.listdir(aishell_dir):
        # if  os.path.isdir(sub_dir):
        print(sub_dir)
        abs_sub_dir = os.path.join(aishell_dir, sub_dir)
        tim = 0
        for f in os.listdir(abs_sub_dir):
            assert os.path.isdir(f) is False
            if 'G' in f:
                t = np.load(os.path.join(abs_sub_dir, f))
                speaker_emb.append(t)
                speaker_label.append(speaker_id)
                tim += 1
                if tim >= N:
                    break
        
        speaker_id += 1
        # break
    

    # Cross-Samples
    for i in range(len(cross_samples_path)):
        path = cross_samples_path[i]
        now_spk_id = cross_samples_speaker_id[i]
        t = np.load(path)
        test_speaker_emb.append(t)
        test_speaker_label.append(now_spk_id)





    speaker_emb = np.asarray(speaker_emb)
    speaker_label = np.asarray(speaker_label)

    test_speaker_emb = np.asarray(test_speaker_emb)
    test_speaker_label = np.asarray(test_speaker_label)

    print(speaker_label)
    print(speaker_emb)
    print(speaker_emb.shape)

    return speaker_emb, speaker_label, test_speaker_emb, test_speaker_label


    




if __name__ == '__main__':
    speaker_emb, speaker_label, test_speaker_emb, test_speaker_label = data_call()
    tsne_plotter(data = speaker_emb, label = speaker_label,
                 test_data = test_speaker_emb, test_label = test_speaker_label,
                 save_png = png_path, title = 'PCA')