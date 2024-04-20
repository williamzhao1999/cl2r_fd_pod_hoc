import numpy as np
import os.path as osp
from sklearn.model_selection import KFold
import torch

from cl2r.utils import extract_features
from cl2r.model import ResNet32Cifar
from cl2r.metrics import backward_compatibility, forward_compatibility
from cl2r.performance_metrics import identification
from cl2r.compatibility_metrics import average_compatibility, average_accuracy
import logging
import faiss
logger = logging.getLogger('Eval')

def evaluate(args, query_loader, gallery_loader):

    compatibility_matrix = np.zeros((args.nb_tasks, args.nb_tasks))
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets

    for task_id in range(args.nb_tasks):
        ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt")) 
        net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device,
                                         args=args)
        net.eval() 
        query_feat = extract_features(args, net, query_loader)

        for i in range(task_id+1):
            ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{i}.pt")) 
            previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device,
                                         args=args)
            previous_net.eval() 
        
            gallery_feat = extract_features(args, previous_net, gallery_loader)
            #acc = verification(query_feat, gallery_feat, targets)
            acc = identification(gallery_feat, gallery_targets, 
                                 query_feat, targets, 
                                 topk=1
                                )
            compatibility_matrix[task_id][i] = acc

            if i != task_id:
                acc_str = f'Cross-test accuracy between model at task {task_id+1} and {i+1}:'
            else:
                acc_str = f'Self-test of model at task {i+1}:'
            print(f'{acc_str} {acc*100:.2f}')

    # compatibility metrics
    ac = average_compatibility(matrix=compatibility_matrix)
    bc = backward_compatibility(matrix=compatibility_matrix)
    fc = forward_compatibility(matrix=compatibility_matrix)

    print(f"Avg. Comp. {ac:.2f}")
    print(f"Backw. Comp. {bc:.3f}")
    print(f"Forw. Comp. {fc:.3f}")

    print(f"Compatibility Matrix:\n{compatibility_matrix}")
    np.save(osp.join(f"./{args.checkpoint_path}/compatibility_matrix.npy"), compatibility_matrix)


def validation(args, net, query_loader, gallery_loader, task_id, selftest=False):
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets

    net.eval() 
    query_feat = extract_features(args, net, query_loader)

    if selftest:
        previous_net = net
    else:
        ckpt_path = osp.join(*(args.root_folder, "checkpoints", f"ckpt_{task_id-1}.pt")) 
        previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                        starting_classes=100, 
                                        feat_size=99, 
                                        device=args.device,
                                        args=args)
        previous_net.eval()
        previous_net.to(args.device)
    gallery_feat = extract_features(args, previous_net, gallery_loader)
    #acc = verification(query_feat, gallery_feat, targets)
    acc = identification(gallery_feat, gallery_targets, 
                                 query_feat, targets, 
                                 topk=1
                                )
    print(f"{'Self' if selftest else 'Cross'} Compatibility Accuracy: {acc*100:.2f}")
    return acc


#"""From [insightface](https://github.com/deepinsight/insightface)"""
#def verification(query_feature, gallery_feature, targets):
#    thresholds = np.arange(0, 4, 0.001)
#    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, query_feature, gallery_feature, targets)
#    return accuracy.mean()

'''
def image2template_feature(img_feats=None,  # features of all images
                           templates=None,  # target of features in input 
                          ):
    
    unique_templates = np.unique(templates)
    unique_subjectids = None

    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        ind_t = np.where(templates == uqt)[0]
        face_norm_feats = img_feats[ind_t]
        template_feats[count_template] = np.mean(face_norm_feats, 0)
        
    logger.info(f'Finish Calculating {count_template} template features.')
    template_norm_feats = template_feats / np.sqrt(
        np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids

def calculate_rank(query_feats, gallery_feats, topk):
    logger.info(f"query_feats shape: {query_feats.shape}")
    logger.info(f"gallery_feats shape: {gallery_feats.shape}")
    num_q, feat_dim = query_feats.shape

    logger.info("=> build faiss index")
    faiss_index = faiss.IndexFlatIP(feat_dim)
    faiss_index.add(gallery_feats)
    logger.info("=> begin faiss search")
    _, ranked_gallery_indices = faiss_index.search(query_feats, topk)
    return ranked_gallery_indices    

def calculate_mAP_gldv2(ranked_gallery_indices, query_gts, topk):
    num_q = ranked_gallery_indices.shape[0]
    average_precision = np.zeros(num_q, dtype=float)
    for i in range(num_q):
        retrieved_indices = np.where(np.in1d(ranked_gallery_indices[i], np.array(query_gts[i])))[0]
        if retrieved_indices.shape[0] > 0:
            retrieved_indices = np.sort(retrieved_indices)
            gts_all_count = min(len(query_gts[i]), topk)
            for j, index in enumerate(retrieved_indices):
                average_precision[i] += (j + 1) * 1.0 / (index + 1)
            average_precision[i] /= gts_all_count
    return np.mean(average_precision)

def identification(gallery_feats, gallery_gts, query_feats, query_gts, topk=1):
    # https://github.com/TencentARC/OpenCompatible/blob/master/data_loader/GLDv2.py#L129

    # check if torch, if yes convert to numpy
    if isinstance(query_feats, torch.Tensor):
        query_feats = query_feats.cpu().numpy()
    if isinstance(gallery_feats, torch.Tensor):
        gallery_feats = gallery_feats.cpu().numpy()
    if isinstance(query_gts, torch.Tensor):
        query_gts = query_gts.cpu().numpy()
    if isinstance(gallery_gts, torch.Tensor):
        gallery_gts = gallery_gts.cpu().numpy()

    query_gts = np.array(query_gts).reshape(-1, 1)
    
    unique_gallery_feats, _, _ = image2template_feature(gallery_feats, 
                                                        gallery_gts)
    unique_gallery_feats = unique_gallery_feats.astype(np.float32)

    logger.info("=> calculate rank")
    ranked_gallery_indices = calculate_rank(query_feats, unique_gallery_feats, topk=1)
    logger.info("=> calculate 1:N search acc")
    mAP = calculate_mAP_gldv2(ranked_gallery_indices, query_gts, topk=1)
    logger.info(f"1:N search acc: {mAP:.4f}")
    return mAP


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds = 10, pca = 0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits = nrof_folds, shuffle = False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                actual_issame[test_set])

        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc
'''

