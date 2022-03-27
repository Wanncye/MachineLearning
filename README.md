# README
>我发现只看机器学习原理，存在两个问题：
* 1.有些算法的原理难以理解
* 2.就算理解了，也不知道实现思路，只能当个调包侠  
>因此，决定借鉴一下别人实现的代码，看看具体的实现思想，也方便更加深入的理解这些机器学习算法。所以，这个仓库适合对下列算法有一定基础了解的，但是对代码实现不清楚的朋友。
其中包括：
* 1.GBDT算法:[https://github.com/Freemanzxp/GBDT_Simple_Tutorial](https://github.com/Freemanzxp/GBDT_Simple_Tutorial)
>如何做特征选择？如何做分类？
* 2.DecisionTree:[https://github.com/lucksd356/DecisionTrees](https://github.com/lucksd356/DecisionTrees)
>只实现了ID3算法，其他两种C4.5，CART在计算增益这一块用的是不同的指标。
这里列出三种决策树的[区别](https://blog.csdn.net/qq_27717921/article/details/74784400)
* 3.XgBoost:[https://blog.csdn.net/slx_share/article/details/82389343](代码来自这个博客)，公式推导我觉得讲的最详细的是[这篇文章](https://zhuanlan.zhihu.com/p/92837676)。
>Xgboost的目标函数、公式推导、GBDT与Xgboost的区别、Xgboost的正则化原理，这些都是需要掌握的问题。
* 4.RandomForest:[https://github.com/zhaoxingfeng/RandomForest](https://github.com/zhaoxingfeng/RandomForest)
>RF为什么要随机抽样？又为什么做有放回的采样？
* 5.AdaBoost:[https://github.com/jaimeps/adaboost-implementation/tree/master](https://github.com/jaimeps/adaboost-implementation/tree/master)
>权值更新的方法、为什么能快速收敛、优缺点
* 6.SVM:[https://vimsky.com/article/222.html](https://vimsky.com/article/222.html),实现的是HingeLoss版本的SVM
>什么叫硬间隔？什么叫软间隔？SVM为什么采用间隔最大化？为什么使用核函数？
* 7.MLE:[https://blog.csdn.net/pengjian444/article/details/71215965](https://blog.csdn.net/pengjian444/article/details/71215965)
>MLE、MAP、贝叶斯估计之间的区别与联系，说老实话，贝叶斯估计没看懂
* 8.GMM(EM):[https://github.com/SmallVagetable/machine_learning_python](https://github.com/SmallVagetable/machine_learning_python)
>这个给的链接库其实已经有好多写好了的算法，也是可以借鉴的。E-step：在已知均值和方差的情况下，判断样本来自第K个模型的概率；M-step：在得知样本来自哪个模型之后，可以通过MLE来估计高斯分布的均值和方差。如此循环，直至收敛。
* 9.LDA:[https://github.com/heucoder/dimensionality_reduction_alo_codes](https://github.com/heucoder/dimensionality_reduction_alo_codes)
>PCA和LDA数据假设都符合高斯分布，但是LDA是监督算法，而PCA是无监督算法。LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。具体LDA原理以及与PCA的区别见[链接](https://www.cnblogs.com/pinard/p/6244265.html)
* 10.PCA(和LDA来自同一个git主)
>从两个算法（PCA和LDA）的流程看，PCA和LDA很相似，只是LDA是加入了标签信息，计算了类内方差，内间均值，然后对这两个数的乘积求特征值、特征向量；而PCA则直接求数据之间的协方差的特征值和特征向量。共同的做法就是取前k个特征值、特征向量，将样本投影到这歌空间中去（包括SVD也是酱紫）。
* TODO
* 10.KNN
* 11.K-Mean
* --T-SNE

