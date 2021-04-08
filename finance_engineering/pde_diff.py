# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:26:15 2021

@author: Lenovo
"""






class call_option_pricing():
    def __init__(self,K,sig_fun,R=0.03,T=1,X=None):
        if X is None:
            X=7*K
        self.K=K
        self.sig_fun=sig_fun
        self.R=R
        self.T=T
        self.X=X
        self.ax=None
    def explicit_diff(self,DX=0.01,DT=0.01):
        import numpy as np
        K=self.K
        sig_fun=self.sig_fun
        R=self.R
        T=self.T 
        X=self.X
        a_fun=lambda x,t:0.5*DT*(x**2*sig_fun(x,t)**2/DX**2-R*x/DX)
        b_fun=lambda x,t:1-DT*(x**2*sig_fun(x,t)**2/DX**2+R)
        c_fun=lambda x,t:0.5*DT*(x**2*sig_fun(x,t)**2/DX**2+R*x/DX)
        x=np.linspace(0,X,int(X/DX+1))
        t=np.linspace(0,T,int(T/DT+1))
        grid_x,grid_t=np.meshgrid(x,t)
        terminal_func=lambda x: np.maximum(x-K,0)
        boundary0_func=lambda t: [0]*len(t)
        boundaryX_func=lambda t: X-K*np.exp(-R*(T-t))
        V=np.zeros((len(x),len(t)))
        V.fill(np.nan)
        V[:,-1]=terminal_func(x)#终止条件
        V[0,:]=boundary0_func(t)#终止条件0
        V[-1,:]=boundaryX_func(t)#终止条件 ∞
        for n in range(1,len(t)):
            for j in range(1,len(x)-1):
                V[j,-n-1]=a_fun(x[j],t[-n])*V[j-1,-n]+b_fun(x[j],t[-n])*V[j,-n]+c_fun(x[j],t[-n])*V[j+1,-n]
        self.x=x
        self.t=t
        self.grid_x=grid_x
        self.grid_t=grid_t
        self.V=V
        self.DX=DX
        
        
        
    def implicit_diff(self,DX=0.01,DT=0.01):
        import numpy as np
        K=self.K
        sig_fun=self.sig_fun
        R=self.R
        T=self.T 
        X=self.X
        def Thomas(La,Mb,Uc,Rd):
            #Mb是矩阵的主对角线元素，长度n。
            #La、Uc是矩阵的下对角线，长度n-1.
            #Rd是方程的解，长度n。
            n=len(Mb)
            c=np.zeros(n-1)
            d=np.zeros(n)
            c[0]=Uc[0]/Mb[0]
            for i in range(1,n-1):
                c[i]=Uc[i]/(Mb[i]-c[i-1]*La[i-1])
            d[0]=Uc[0]/Mb[0]
            for i in range(1,n):
                d[i]=(Rd[i]-(d[i-1]*La[i-1]))/(Mb[i]-c[i-1]*La[i-1])
            ls=list(range(n-1))[::-1]
            x=np.zeros(n)
            x[n-1]=d[n-1]
            for i in ls:
                x[i]=d[i]-c[i]*x[i+1]
            return(x)

        a_fun=lambda x,t:-0.5*DT*(x**2*sig_fun(x,t)**2/DX**2-R*x/DX)
        b_fun=lambda x,t:1+DT*(x**2*sig_fun(x,t)**2/DX**2+R)
        c_fun=lambda x,t:-0.5*DT*(x**2*sig_fun(x,t)**2/DX**2+R*x/DX)
        
        x=np.linspace(0,X,int(X/DX+1))
        t=np.linspace(0,T,int(T/DT+1))
        grid_x,grid_t=np.meshgrid(x,t)
        
        terminal_func=lambda x: np.maximum(x-K,0)
        boundary0_func=lambda t: [0]*len(t)
        boundaryX_func=lambda t: X-K*np.exp(-R*(T-t))
        V=np.zeros((len(x),len(t)))
        V.fill(np.nan)
        V[:,-1]=terminal_func(x)#终止条件
        V[0,:]=boundary0_func(t)#终止条件0
        V[-1,:]=boundaryX_func(t)#终止条件 ∞
        
        for n in range(1,len(t)):
            M_L=[]
            U_L=[]
            L_L=[]
            for j in range(1,len(x)-1):
                M_L.append(b_fun(x[j],t[-n-1]))
                U_L.append(c_fun(x[j],t[-n-1]))
                L_L.append(a_fun(x[j],t[-n-1]))
            M=[1]+M_L+[1]
            U=[0]+U_L
            L=L_L+[0]
            d=[V[0,-n-1]]+V[1:-1,-n].tolist()+[V[-1,-n-1]]
            V[:,-n-1]=Thomas(L,M,U,d)
        self.x=x
        self.t=t
        self.grid_x=grid_x
        self.grid_t=grid_t
        self.V=V
        self.DX=DX
    def crank_nicolson(self,DX=0.01,DT=0.01):
        import numpy as np
        K=self.K
        sig_fun=self.sig_fun
        R=self.R
        T=self.T 
        X=self.X
        def Thomas(La,Mb,Uc,Rd):
            #Mb是矩阵的主对角线元素，长度n。
            #La、Uc是矩阵的下对角线，长度n-1.
            #Rd是方程的解，长度n。
            n=len(Mb)
            c=np.zeros(n-1)
            d=np.zeros(n)
            c[0]=Uc[0]/Mb[0]
            for i in range(1,n-1):
                c[i]=Uc[i]/(Mb[i]-c[i-1]*La[i-1])
            d[0]=Uc[0]/Mb[0]
            for i in range(1,n):
                d[i]=(Rd[i]-(d[i-1]*La[i-1]))/(Mb[i]-c[i-1]*La[i-1])
            ls=list(range(n-1))[::-1]
            x=np.zeros(n)
            x[n-1]=d[n-1]
            for i in ls:
                x[i]=d[i]-c[i]*x[i+1]
            return(x)
        
        a_fun=lambda x,t:0.25*DT*(x**2*sig_fun(x,t)**2/DX**2-R*x/DX)
        b_fun=lambda x,t:-1-0.5*DT*(x**2*sig_fun(x,t)**2/DX**2+R)
        c_fun=lambda x,t:0.25*DT*(x**2*sig_fun(x,t)**2/DX**2+R*x/DX)
        bs_fun=lambda x,t:-1+0.5*DT*(x**2*sig_fun(x,t)**2/DX**2+R)
        
        x=np.linspace(0,X,int(X/DX+1))
        t=np.linspace(0,T,int(T/DT+1))
        grid_x,grid_t=np.meshgrid(x,t)
        
        terminal_func=lambda x: np.maximum(x-K,0)
        boundary0_func=lambda t: [0]*len(t)
        boundaryX_func=lambda t: X-K*np.exp(-R*(T-t))
        V=np.zeros((len(x),len(t)))
        V.fill(np.nan)
        V[:,-1]=terminal_func(x)#终止条件
        V[0,:]=boundary0_func(t)#终止条件0
        V[-1,:]=boundaryX_func(t)#终止条件 ∞
        
        for n in range(1,len(t)):
            M_L=[]
            U_L=[]
            L_L=[]
            d_L=[]
            for j in range(1,len(x)-1):
                M_L.append(b_fun(x[j],t[-n-1]))
                U_L.append(c_fun(x[j],t[-n-1]))
                L_L.append(a_fun(x[j],t[-n-1]))
                d_L.append(-a_fun(x[j],t[-n])*V[j-1,-n]+\
                           bs_fun(x[j],t[-n])*V[j,-n]-\
                           c_fun(x[j],t[-n])*V[j+1,-n] )
            M=[1]+M_L+[1]
            U=[0]+U_L
            L=L_L+[0]
            d=[V[0,-n-1]]+d_L+[V[-1,-n-1]]
            V[:,-n-1]=Thomas(L,M,U,d)
            
        self.x=x
        self.t=t
        self.grid_x=grid_x
        self.grid_t=grid_t
        self.V=V 
        self.DX=DX        
    def interp(self):
        from scipy.interpolate import interp2d
        return interp2d(self.t, self.x, self.V, kind='cubic')

    def plot_grid(self,axis=None,c='g',return_=True):
        import matplotlib.pyplot as plt 
        grid_x = self.grid_x
        grid_t = self.grid_t      
        V=self.V
        DX=self.DX
        if axis is None:
            ax = plt.axes(projection='3d')
            ax.plot_wireframe(grid_x.T[:int(2*self.K/DX+1),:], \
                              grid_t.T[:int(2*self.K/DX+1),:], \
                              V[:int(2*self.K/DX+1),:],color=c)
            ax.set_title('surface')
            ax.view_init(50, 100)
            if return_:
                return ax
        else:            
            axis.plot_wireframe(grid_x.T[:int(2*self.K/DX+1),:], \
                              grid_t.T[:int(2*self.K/DX+1),:], \
                              V[:int(2*self.K/DX+1),:],color=c)
        

    def B_S_M_formula(self,time,price,sigma):
        
        def vanilla_option(t, S,K, T, r, sigma, option='call'):  
            from scipy.stats import norm
            import numpy as np
            tau=T-t
            if tau >0:
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
                d2 = (np.log(S/K) + (r - 0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
                if option == 'call':
                    p = (S*norm.cdf(d1, 0.0, 1.0) - K*np.exp(-r*tau)*norm.cdf(d2, 0.0, 1.0))
                elif option == 'put':
                    p = (K*np.exp(-r*tau)*norm.cdf(-d2, 0.0, 1.0) - S*norm.cdf(-d1, 0.0, 1.0))
                else:
                    return None
            else:
                if option == 'call':
                    p = max(S-K,0)
                elif option == 'put':
                    p = max(K-S,0)
                else:
                    return None        
            return p
        return vanilla_option(time,price, self.K,self.T, self.R, sigma, option='call')


    

if __name__ == '__main__':
    from finance_engineering.pde_diff import call_option_pricing
    sig_fun=lambda x,t:0.1*x**-0.9
    ca=call_option_pricing(2,sig_fun)
    ca.crank_nicolson()
    ax=ca.plot_grid()

    sig_fun=lambda x,t:0.9*x**-0.1
    ca=call_option_pricing(2,sig_fun)
    ca.crank_nicolson()
    ca.plot_grid(axis=ax,c='r')

    sig_fun=lambda x,t:1
    ca=call_option_pricing(2,sig_fun)
    ca.crank_nicolson()
    f=ca.interp()
    print(f(0,2)[0])
    print(ca.B_S_M_formula(0,2,1))





