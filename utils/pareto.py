import numpy as np
import torch
import torch.nn.functional as F

class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.85#99
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.15 #0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps


    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        return sol_vec , init_sol[2]

    # def find_min_norm_element_init(vecs):
    #    vecs_mean = torch.mean(vecs.float(),dim=0)
    # #    print('vecs_mean', vecs_mean)
    #    d_min, min_idx = 1e5,0
    #    N = len(vecs)
    #    for i in range(len(vecs)):
    #        inner_prod = (vecs[i]*vecs_mean).sum()
    #     #    print('inner_prod',inner_prod)
    #        if inner_prod < d_min:
    #            d_min = inner_prod
    #            min_idx = i
    # #    print('min_idx',min_idx)
    #    vecs_new = []
    #    vecs_new.append(vecs_mean)
    #    vecs_new.append(vecs[min_idx])
    # #    print('vecs_new',vecs_new)
    #    sol_vec, init_sol =  MinNormSolver.find_min_norm_element(vecs_new)
    #    return sol_vec[0]*np.array([1/N]*N) + sol_vec[1]*np.eye(N)[min_idx]

# single query
    # def find_min_norm_element_init(vecs, weights):
    #     # vecs_mean = torch.mean(vecs.float(),dim=0)
    #     vecs_mean = (vecs*weights).sum()
    #     d_min, min_idx = 1e5,0
    #     N = len(vecs)
    #     for i in range(len(vecs)):
    #         inner_prod = vecs[i]*vecs_mean 
    #         if inner_prod < d_min:
    #             d_min = inner_prod
    #             min_idx = i

    #     vecs_new = []
    #     vecs_new.append(vecs_mean)
    #     vecs_new.append(vecs[min_idx])
    #     sol_vec, init_sol =  MinNormSolver.find_min_norm_element(vecs_new)

    #     weights = torch.tensor(sol_vec[0]*weights) + torch.tensor(sol_vec[1]*np.eye(N)[min_idx]).cuda()
    #     return weights/weights.sum() 


    # mutli-maml (version-1)
    def find_min_norm_element_init(vecs, weights):
        # vecs_mean = torch.mean(vecs.float(),dim=0)
        vecs = vecs[:,-3210:]
        # vecs = vecs[:,-5000:-3210]
        vecs_mean = (vecs*weights.unsqueeze(1)).sum(0)
        # vecs_mean = vecs[-1]
        d_min, min_idx = 1e5,0
        N = len(vecs)
        for i in range(len(vecs)):
            # inner_prod = (vecs[i]*vecs_mean).sum()
            # print('vecs0', vecs[i].shape)
            # print('vces1', vecs_mean.shape)
            inner_prod = F.cosine_similarity(vecs[i].unsqueeze(0), vecs_mean.unsqueeze(0))
            # print('i,inner_prod', i, inner_prod)
            if inner_prod < d_min:
                d_min = inner_prod
                min_idx = i
        # print('min_idx', min_idx)
        # if d_min >0:
        #     return weights*0+torch.tensor(np.eye(N)[N-1]).cuda()
        # else:
        vecs_new = []
        vecs_new.append(vecs_mean)
        vecs_new.append(vecs[min_idx])
        # print('vecs_new', torch.stack(vecs_new).shape)
        sol_vec, init_sol =  MinNormSolver.find_min_norm_element(torch.stack(vecs_new))
        # print('weights', weights)
        # print('0', sol_vec[0] )
        # print('1', sol_vec[1], min_idx, np.eye(N)[min_idx] )
        weights = torch.tensor(sol_vec[0]*weights) + torch.tensor((1-sol_vec[0])*np.eye(N)[min_idx]).cuda()
        # print('weights0',weights, weights[-1])
        # weights[-1] = (weights[-1]+1).clone()
        # print('weights',weights)
        return weights#/weights.sum() 

    # mutli-maml (version-2)
    # def find_min_norm_element_init(vecs, weights):
    #     vecs_mean = vecs[-1] + (vecs[:-1]*weights[:-1].unsqueeze(1)).sum(0)

    #     d_min, min_idx = 1e5,0
    #     N = len(vecs[:-1])
    #     for i in range(N):
    #         inner_prod = (vecs[i]*vecs_mean).sum()
    #         if inner_prod < d_min:
    #             d_min = inner_prod
    #             min_idx = i
    #     print('min_idx', min_idx)
    #     print('weights', weights[min_idx])
        
    #     temp_vecs =  vecs_mean - weights[min_idx]*vecs[min_idx]
    #     print('temp', temp_vecs,-torch.matmul(temp_vecs, vecs[min_idx].unsqueeze(0).T))
    #     print('1',torch.matmul(vecs[min_idx].unsqueeze(0), vecs[min_idx].unsqueeze(0).T))
    #     weights_up = max(0, -torch.matmul(temp_vecs, vecs[min_idx].unsqueeze(0).T)/( torch.matmul(vecs[min_idx].unsqueeze(0), vecs[min_idx].unsqueeze(0).T) ) )
    #     weights_up = min(weights_up, 10)

    #     weights[min_idx] = weights_up
    #     return weights


# sorted
    # def find_min_norm_element_init(vecs, weights):
    #     # vecs_mean = torch.mean(vecs.float(),dim=0)
    #     # take the order into account
    #     temp = torch.mean(vecs[:,:-1].float(),dim=1) # according to the performance of previous tasks
    #     sorted_index = torch.argsort(temp, descending=True)
    #     # print('vecs0', vecs)
    #     vecs = vecs[sorted_index]
    #     # print('vecs', vecs)
    #     # print('weights',weights)
    #     vecs_mean = (vecs*weights.unsqueeze(1)).sum(0)
        
    #     d_min, min_idx = 1e5,0
    #     N = len(vecs)
    #     for i in range(len(vecs)):
    #         inner_prod = (vecs[i]*vecs_mean).sum()
    #         if inner_prod < d_min:
    #             d_min = inner_prod
    #             min_idx = i
    #     vecs_new = []
    #     vecs_new.append(vecs_mean)
    #     vecs_new.append(vecs[min_idx])
    #     sol_vec, init_sol =  MinNormSolver.find_min_norm_element(vecs_new)
        
    #     sol_vec[0] = 0.1 if sol_vec[0]<0.1 else sol_vec[0]
    #     sol_vec[0] = 0.9 if sol_vec[0]>0.9 else sol_vec[0]
    #     print('sol_vec',sol_vec)
    #     weights = torch.tensor(sol_vec[0]*weights) + torch.tensor((1-sol_vec[0])*np.eye(N)[min_idx]).cuda() 
    #     # resort_weights = torch.zeros_like(weights)
    #     # resort_weights[sorted_index] = weights
    #     return  sorted_index, weights             

if __name__ == '__main__':
    vecs = torch.tensor([[-22, -3, 3],
            [2,10,14],
            [-5,-8,7],
            ])
    a = MinNormSolver.find_min_norm_element_init(vecs)