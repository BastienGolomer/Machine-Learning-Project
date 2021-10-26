def PCA_min(nb_col_max):
    ordered_matrix=[]
    data_copy=data.copy()
    for i in range 1,nb_col_max):
        #takes columns from 1 to nb_max_col
        tempory_loss=[10**1000,]
        for j in range (0,len(data_copy)-1):
            loss=compute_loss(y,np.c_(ordered_matrix,datacopy[:,j]) #jsp si ca marche
            if loss<tempory_loss[0]:
                tempory_loss=[loss, datacopy[:,j]]
        np.c_(ordered_matrix, tempory_loss[1])
        data_copy.rm(tempory_loss[1])
        losses.append(loss[0])
    return ordered ordered_matrix,losses

def PCA_min(nb_col_min):
    ordered_matrix=[]
    data_copy=data.copy()
    for i in range 1,len(data[0,:]):
        tempory_loss=[10**1000,]
        for j in range (0,len(data_copy)-1):
            loss=compute_loss(y,np.c_(ordered_matrix,datacopy[:,j]) #jsp si ca marche
            if loss<tempory_loss[0]:
                tempory_loss=[loss, datacopy[:,j]]
        np.c_(ordered_matrix, tempory_loss[1])
        data_copy.rm(tempory_loss[1])
        losses.append(loss[0])
    return ordered ordered_matrix,losses

#k-fold cross validation?
#knn?
#how to visualise if the data is flexible or unflexible?