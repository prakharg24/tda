import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import pickle


# new_w = np.load("final_clusters.npy", allow_pickle=True)
#
# for ele in new_w:
#     print(ele)
#
# exit()


comp_w = np.load("early_weights.npy", allow_pickle=True)

# for ele in comp_w:
#     print(np.shape(ele))

# clusters = np.load("final_clusters2.npy", allow_pickle=True)
# clusters = [x for x in clusters]

# print(len(clusters))

# clusters.append(np.zeros((1)))
# np.save("final_clusters2", np.array(clusters))
# exit()

# new_clusters = []
# i=0
# j=0
# while(i<len(clusters)):
#     if(len(comp_w[j].shape)==1):
#         new_clusters.append(np.zeros(comp_w[j].shape))
#         j += 1
#     else:
#         new_clusters.append(clusters[i])
#         i += 1
#         j += 1
#
# for ele in new_clusters:
#     print(ele.shape)

# np.save("final_clusters2", np.array(new_clusters))
#
# exit()
clusters = []


# file = open("final_clusters.pkl",'rb')
# clusters = pickle.load(file)
# file.close()
# print(clusters)

print(len(clusters))

def takeSecond(elem):
    return elem[1]

for i, ele in enumerate(comp_w):
    # if(i<18):
    #     continue
    # if(len(ele.shape)==1):
    #     continue
    print("Layer ", i)
    st = time.time()
    nzs = np.nonzero(ele)
    to_cls = ele[nzs]
    print(np.shape(ele), np.shape(nzs), np.shape(to_cls))

    combine = [(a, b) for a, b in zip(np.transpose(nzs), to_cls)]

    combine.sort(key=takeSecond)

    nzs = np.array([a for (a, b) in combine])
    to_cls = np.array([b for (a, b) in combine])

    end = 0
    final_cls = []
    sum_k = 0
    while (end<len(to_cls)):
        print("Partition :", end)
        curr_arr = to_cls[end:min(end+1000, len(to_cls))]
        curr_arr = curr_arr.reshape((-1, 1))
        k = max(1, curr_arr.size//5)
        sum_k += k
        kmeans = KMeans(n_clusters=k, verbose=0).fit(curr_arr)
        ans = kmeans.labels_

        for cl_ind in range(np.amax(ans)+1):
            curr_cls = [a for (a, b) in zip(nzs[end:min(end+1000, len(to_cls))], ans) if(b==cl_ind)]
            final_cls.append(curr_cls)

        end = min(end+1000, len(to_cls))

    print(sum_k, np.shape(final_cls), time.time() - st)
    # for ele in final_cls:
    #     print(np.shape(ele))
    clusters.append(final_cls)

    filehandler = open("final_clusters.pkl","wb")
    pickle.dump(clusters,filehandler)
    filehandler.close()
