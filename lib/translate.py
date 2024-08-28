import torch 
import numpy as np



class translate2D(torch.autograd.Function):
    
    
    @staticmethod
    def forward(ctx, input):
        u,v  = input
        
#         def f(x,y):
#             return np.exp(-((x-u)**2/(2*0.5)+(y-v)**2/(2*0.5)))
 
        
#         for i in range(16):
#             for j in range(16):
#                 a[i,j]= f(i,j)

        
        
        # x = torch.arange(128).to("cuda:0")
        # y = torch.arange(128).to("cuda:0")
        x = torch.arange(128)
        y = torch.arange(128)
        
        # grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # grid_x = grid_x.to("cuda:0")
        # grid_y = grid_y.to("cuda:0")
        a = torch.exp(-((grid_x-u)**2/(2*50)+(grid_y-v)**2/(2*50)))
        
        ctx.save_for_backward(input)
        return a.flatten()

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        u,v = input
        
        # g =  torch.zeros((2,128,128)).float().to("cuda:0")
        g =  torch.zeros((2,128,128)).float()
           
#         for i in range(16):
#             for j in range(16):
#                 g[0,i,j] =  (-u + i)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
#                 g[1,i,j] =  (-v + j)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
        
    
    
        x = torch.arange(128)
        y = torch.arange(128)
        # x = torch.arange(128).to("cuda:0")
        # y = torch.arange(128).to("cuda:0")
        
        
        grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
        # grid_x = grid_x.to("cuda:0")
        # grid_y = grid_y.to("cuda:0")

        
        g[0] = (-u + grid_x)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
        g[1] = (-v + grid_y)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
        
      
        ret = torch.reshape(g, (2,-1))@grad_output
  
        return ret





# class translate2D_cuda(torch.autograd.Function):
    
    
#     @staticmethod
#     def forward(ctx, input):
#         u,v  = input
        
# #         def f(x,y):
# #             return np.exp(-((x-u)**2/(2*0.5)+(y-v)**2/(2*0.5)))
 
        
# #         for i in range(16):
# #             for j in range(16):
# #                 a[i,j]= f(i,j)

        
        
#         # x = torch.arange(128).to("cuda:0")
#         # y = torch.arange(128).to("cuda:0")
#         x = torch.arange(128).to("cuda:0")
#         y = torch.arange(128).to("cuda:0")
        
#         grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
#         grid_x = grid_x.to("cuda:0")
#         grid_y = grid_y.to("cuda:0")
#         # grid_x = grid_x.to("cuda:0")
#         # grid_y = grid_y.to("cuda:0")
#         a = torch.exp(-((grid_x-u)**2/(2*50)+(grid_y-v)**2/(2*50)))
        
#         ctx.save_for_backward(input)
#         return a.flatten()

#     @staticmethod
#     def backward(ctx, grad_output):

#         input, = ctx.saved_tensors
#         u,v = input
        
#         # g =  torch.zeros((2,128,128)).float().to("cuda:0")
#         g =  torch.zeros((2,128,128)).float().to("cuda:0")
           
# #         for i in range(16):
# #             for j in range(16):
# #                 g[0,i,j] =  (-u + i)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
# #                 g[1,i,j] =  (-v + j)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
        
    
    
#         x = torch.arange(128).to("cuda:0")
#         y = torch.arange(128).to("cuda:0")
#         # x = torch.arange(128).to("cuda:0")
#         # y = torch.arange(128).to("cuda:0")
        
        
#         grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
#         grid_x = grid_x.to("cuda:0")
#         grid_y = grid_y.to("cuda:0")
#         # grid_x = grid_x.to("cuda:0")
#         # grid_y = grid_y.to("cuda:0")

        
#         g[0] = (-u + grid_x)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
#         g[1] = (-v + grid_y)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
#         ret = torch.reshape(g, (2,-1))@grad_output
  
#         return ret    




def gen_point_proj_operator_batched(sigma,size, device = "cpu"):
    class point_projection(torch.autograd.Function):


        @staticmethod
        def forward(ctx, inputs):
            points_pos = inputs

           
    #         def f(x,y):
    #             return np.exp(-((x-u)**2/(2*0.5)+(y-v)**2/(2*0.5)))


    #         for i in range(16):
    #             for j in range(16):
    #                 a[i,j]= f(i,j)



            # x = torch.arange(128).to("cuda:0")
            # y = torch.arange(128).to("cuda:0")
            x = torch.arange(size).to(device)
            # y = torch.arange(128)

            # grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
            grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')

            # grid_x = grid_x.to(device)
            # grid_y = grid_y.to(device)
   
            c1 = points_pos[:,:,0].reshape((len(points_pos),100,1,1))
            c2 = points_pos[:,:,1].reshape((len(points_pos),100,1,1))
            a = torch.sum(torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2))),dim=1)
            # a = torch.exp(-((grid_x-u)**2/(2*50)+(grid_y-v)**2/(2*50)))

            ctx.save_for_backward(points_pos)
            ctx.sigma = sigma


            return torch.flatten(a, start_dim=1)

        @staticmethod
        def backward(ctx, grad_output):


            inputs, = ctx.saved_tensors

            points_pos = inputs
            # sigma = ctx.sigma
            # g =  torch.zeros((2,128,128)).float().to(device)
            g =  torch.zeros((len(points_pos),2,100,size,size)).float().to(device)

    #         for i in range(16):
    #             for j in range(16):
    #                 g[0,i,j] =  (-u + i)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
    #                 g[1,i,j] =  (-v + j)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))


            
            x = torch.arange(size).to(device)
            y = torch.arange(size).to(device)
            # x = torch.arange(128).to(device)
            # y = torch.arange(128).to(device)


            grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
            # grid_x = grid_x.to(device)
            # grid_y = grid_y.to(device)
            # c1 = points_pos[:,0].reshape((len(points_pos),100,1,1))
            c1 = points_pos[:,:,0].reshape((len(points_pos),100,1,1))
            c2 = points_pos[:,:,1].reshape((len(points_pos),100,1,1))

            g[:,0] =  (-c1 + grid_x) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))
            g[:,1] = (-c2 + grid_y) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))

    #         g[0] = (-u + grid_x)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
    #         g[1] = (-v + grid_y)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))

          
            # ret = torch.bmm(torch.reshape(g, (len(points_pos),2,100,-1)),grad_output).T
           
            ret = torch.einsum('baij,bj->bai', torch.reshape(g, (len(points_pos),2,100,-1)), grad_output).permute(0,2,1)
            
            return ret.clone()
    return  point_projection.apply




def gen_point_proj_operator(sigma,size, device = "cpu"):
    class point_projection(torch.autograd.Function):


        @staticmethod
        def forward(ctx, inputs):
            points_pos = inputs

           
    #         def f(x,y):
    #             return np.exp(-((x-u)**2/(2*0.5)+(y-v)**2/(2*0.5)))


    #         for i in range(16):
    #             for j in range(16):
    #                 a[i,j]= f(i,j)



            # x = torch.arange(128).to("cuda:0")
            # y = torch.arange(128).to("cuda:0")
            x = torch.arange(size).to(device)
            # y = torch.arange(128)

            # grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
            grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')

            # grid_x = grid_x.to("cuda:0")
            # grid_y = grid_y.to("cuda:0")

            c1 = points_pos[:,0].reshape((100,1,1))
            c2 = points_pos[:,1].reshape((100,1,1))
            a = torch.sum(torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2))),dim=0)
            # a = torch.exp(-((grid_x-u)**2/(2*50)+(grid_y-v)**2/(2*50)))

            ctx.save_for_backward(points_pos)
            ctx.sigma = sigma


            return a.flatten()

        @staticmethod
        def backward(ctx, grad_output):


            inputs, = ctx.saved_tensors

            points_pos = inputs
            # sigma = ctx.sigma
            # g =  torch.zeros((2,128,128)).float().to("cuda:0")
            g =  torch.zeros((2,100,size,size)).float().to(device)

    #         for i in range(16):
    #             for j in range(16):
    #                 g[0,i,j] =  (-u + i)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
    #                 g[1,i,j] =  (-v + j)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))



            x = torch.arange(size).to(device)
            y = torch.arange(size).to(device)
            # x = torch.arange(128).to("cuda:0")
            # y = torch.arange(128).to("cuda:0")


            grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
            # grid_x = grid_x.to("cuda:0")
            # grid_y = grid_y.to("cuda:0")

            c1 = points_pos[:,0].reshape((100,1,1))
            c2 = points_pos[:,1].reshape((100,1,1))
            g[0] =  (-c1 + grid_x) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))
            g[1] = (-c2 + grid_y) *torch.exp(-((grid_x-c1)**2/(2*sigma**2)+(grid_y-c2)**2/(2*sigma**2)))

    #         g[0] = (-u + grid_x)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
    #         g[1] = (-v + grid_y)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))


            ret = (torch.reshape(g, (2,100,-1))@grad_output).T

            return ret
    return  point_projection.apply



# class point_projection_cuda(torch.autograd.Function):
    
    
#     @staticmethod
#     def forward(ctx, input):
#         points_pos,  = input
        
# #         def f(x,y):
# #             return np.exp(-((x-u)**2/(2*0.5)+(y-v)**2/(2*0.5)))
 
        
# #         for i in range(16):
# #             for j in range(16):
# #                 a[i,j]= f(i,j)

        
        
#         # x = torch.arange(128).to(device)
#         # y = torch.arange(128).to(device)
#         x = torch.arange(128).to("cuda:0")

        
#         # grid_x, grid_y =torch.meshgrid(x, y, indexing='ij')
#         grid_x, grid_y = torch.meshgrid(x, x, indexing='ij')

#         grid_x = grid_x.to("cuda:0")
#         grid_y = grid_y.to("cuda:0")

#         c1 = points_pos[:,0].reshape((100,1,1))
#         c2 = points_pos[:,1].reshape((100,1,1))
#         a = torch.sum(torch.exp(-((grid_x-c1)**2/(2*50)+(grid_y-c2)**2/(2*50))),dim=0)
#         # a = torch.exp(-((grid_x-u)**2/(2*50)+(grid_y-v)**2/(2*50)))
        
#         ctx.save_for_backward(input)
#         return a.flatten()

#     @staticmethod
#     def backward(ctx, grad_output):

#         input, = ctx.saved_tensors
#         points_pos = input
        
#         # g =  torch.zeros((2,128,128)).float().to("cuda:0")
#         g =  torch.zeros((2,128,128)).float().to("cuda:0")
           
# #         for i in range(16):
# #             for j in range(16):
# #                 g[0,i,j] =  (-u + i)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
# #                 g[1,i,j] =  (-v + j)*np.exp(-(-u + i)**2/(2*0.5) - (-v +j)**2/(2*0.5))
        
    
    
#         x = torch.arange(128)
#         # x = torch.arange(128).to("cuda:0")
#         # y = torch.arange(128).to("cuda:0")
        
        
#         grid_x, grid_y =torch.meshgrid(x, x, indexing='ij')
#         grid_x = grid_x.to("cuda:0")
#         grid_y = grid_y.to("cuda:0")

#         c1 = points_pos[:,0].reshape((100,1,1))
#         c2 = points_pos[:,1].reshape((100,1,1))
#         g[0] =  torch.sum((-c1 + grid_x) * torch.exp(-((grid_x-c1)**2/(2*50)+(grid_y-c2)**2/(2*50))),dim=0)
#         g[1] =  torch.sum((-c2 + grid_y) * torch.exp(-((grid_x-c1)**2/(2*50)+(grid_y-c2)**2/(2*50))),dim=0)
        
# #         g[0] = (-u + grid_x)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
# #         g[1] = (-v + grid_y)*torch.exp(-(-u + grid_x)**2/(2*50) - (-v + grid_y)**2/(2*50))
        
        
#         ret = torch.reshape(g, (2,-1))@grad_output
  
#         return ret