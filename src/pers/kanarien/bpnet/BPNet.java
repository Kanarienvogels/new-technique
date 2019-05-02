package pers.kanarien.bpnet;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

/**
 * BP网络
 * 关键点：
 * 1. BP网络实质是三层的神经网络，包括输入层、隐层和输出层，其中隐层和输出层都是神经元，有权值，
 *    通过修正权值来建立模型。隐层可以是多层，这里的实现只使用单层
 * 2. 反向传播时采用梯度下降法修正权值，同时采用动量法辅助修正，一定程度上防止陷入局部最优解
 * 3. 激活函数使用Sigmod函数，误差函数使用平方和误差函数
 * @author Kanarien 
 * @version 1.0
 * @date 2019年5月2日 下午12:07:20
 */
public class BPNet {
    private static final double RHO = 0.1;            // 学习速率
    private static final double ALFA = 0.67;          // 动量因子
    private double error;                             // 误差指标 
    
    private int inputLayerNum;    // 输入层神经元数目，对应输入维数
    private int hiddenLayerNum;   // 隐层神经元数目，自定义
    private int outputLayerNum;   // 输出层神经元数目，对应输出维数
    
    private double[][] inWeight;       // 输入层到隐层的权值
    private double[][] outWeight;      // 隐层到输出层的权值
    
    // 以下属性均为了减少重复申请内存而共用的变量
    // 实际上可改为局部变量
    private double[][] dInWeight;      // 输入层到隐层权值的修正值
    private double[][] dOutWeight;     // 隐层到输出层权值的修正值
    private double[][] preInWeight;    // 上次输入层到隐层权值，用于动量计算
    private double[][] prePreInWeight; // 上上次输入层到隐层权值，用于动量计算
    private double[][] preOutWeight;   // 上次隐层到输出层权值，用于动量计算 
    private double[][] prePreOutWeight;// 上上次隐层到输出层权值，用于动量计算
    
    private double[] xi;           // 输入层数据
    private double[] xj;           // 隐层权值和
    private double[] xjActive;     // 隐层权值和激活输出
    private double[] xk;           // 输出层权值和
    private double[] ek;           // 计算输出与教师信号差值
    private double curError = 0.1; // 当前误差，初始为非零值
    
    public BPNet(int inputLayerNum, int hiddenLayerNum, int outputLayerNum) {
        this.inputLayerNum = inputLayerNum;
        this.hiddenLayerNum = hiddenLayerNum;
        this.outputLayerNum = outputLayerNum;
        init();
    }
    
    /**
     * 初始化BP网：
     * 1. 初始化数组
     * 2. 初始化权重
     * 3. 初始化误差参数
     */
    private void init() {
        // 初始化数组
        inWeight = new double[inputLayerNum][hiddenLayerNum];
        outWeight = new double[hiddenLayerNum][outputLayerNum];
        dInWeight = new double[inputLayerNum][hiddenLayerNum];
        dOutWeight = new double[hiddenLayerNum][outputLayerNum];
        preInWeight = new double[inputLayerNum][hiddenLayerNum];
        prePreInWeight = new double[inputLayerNum][hiddenLayerNum];
        preOutWeight = new double[hiddenLayerNum][outputLayerNum];
        prePreOutWeight = new double[hiddenLayerNum][outputLayerNum];
        xi = new double[inputLayerNum];
        xj = new double[hiddenLayerNum];
        xjActive = new double[hiddenLayerNum];
        xk = new double[outputLayerNum];
        ek = new double[outputLayerNum];
        
        // 初始化权重，赋值为-0.5到0.5的一个随机数
        for (int i = 0; i < inputLayerNum; i++) {
            for (int j = 0; j < hiddenLayerNum; j++) {
                inWeight[i][j] = 0.5 - Math.random();
                xj[j] = 0;
            }
        }
        for (int j = 0; j < hiddenLayerNum; j++) {
            for (int k = 0; k < outputLayerNum; k++) {
                outWeight[j][k] = 0.5 - Math.random();
                xk[k] = 0;
            }
        }
        
        // 初始化误差指标
        error = Math.pow(10, -15);
    }
    
    /**
     * BP网络前向处理，每次处理一对输入和输出
     * 输出是理想输出（教师信号）
     * @param input 输入数组
     * @param output 输出数组
     */
    public void forwardProcess(double[] input, double[] output) {
        // 输入层赋值
        for (int i = 0; i < inputLayerNum; i++) {
            xi[i] = input[i];
        }
        // 输入层到隐层权值和计算
        for (int j = 0; j < hiddenLayerNum; j++) {
            xj[j] = 0;
            for (int i = 0; i < inputLayerNum; i++) {
                xj[j] = xj[j] + xi[i] * inWeight[i][j];
            }
        }
        // 隐层使用S函数进行激活
        for (int j = 0; j < hiddenLayerNum; j++) {
            xjActive[j] = 1 / (1 + Math.exp(-xj[j]));
        }
        // 隐层到输出层权值和计算
        for (int k = 0; k < outputLayerNum; k++) {
            xk[k] = 0;
            for (int j = 0; j < hiddenLayerNum; j++) {
                xk[k] = xk[k] + xjActive[j] * outWeight[j][k];
            }
        }
        // 计算输出与教师信号的偏差
        for (int k = 0; k < outputLayerNum; k++) {
            ek[k] = output[k] - xk[k];
        }
        // 计算当前误差
        curError = 0;
        for (int k = 0; k < outputLayerNum; k++) {
            curError = curError + ek[k] * ek[k] / 2.0;  // 误差函数
        }
    }
    
    /**
     * BP网络反向传播，修正权重
     */
    public void backProcess() {
        // 输入层到隐层的权值修正
        for (int i = 0; i < inputLayerNum; i++) {
            for (int j = 0; j < hiddenLayerNum; j++) {
                for (int k = 0; k < outputLayerNum; k++) {
                    dInWeight[i][j] = dInWeight[i][j] + RHO * (ek[k] * xjActive[j] * outWeight[j][k] 
                            * (1 - xjActive[j]) * xi[i]);  // 梯度下降
                }
                inWeight[i][j] = inWeight[i][j] + dInWeight[i][j] + ALFA 
                        * (preInWeight[i][j] - prePreInWeight[i][j]); // 动量调整
                prePreInWeight[i][j] = preInWeight[i][j];
                preInWeight[i][j] = inWeight[i][j];
            }
        }
        // 隐层到输出层的权值修正（梯度下降和动量调整）
        for (int j = 0; j < hiddenLayerNum; j++) {
            for (int k = 0; k < outputLayerNum; k++) {
                dOutWeight[j][k] = RHO * ek[k] * xjActive[j]; // 梯度下降
                outWeight[j][k] = outWeight[j][k] + dOutWeight[j][k] + ALFA 
                        * (preOutWeight[j][k] - prePreOutWeight[j][k]); // 动量调整
                prePreOutWeight[j][k] = preOutWeight[j][k];
                preOutWeight[j][k] = outWeight[j][k];
            }
        }
    }
    
    /**
     * 给出输入，通过BP模型计算出结果
     * @param input 输入
     * @return
     */
    public double[] calculate(double[] input) {
        // 输入层赋值
        for (int i = 0; i < inputLayerNum; i++) {
            xi[i] = input[i]; 
        }
        // 输入层到隐层权值和计算
        for (int j = 0; j < hiddenLayerNum; j++) {
            xj[j] = 0;
            for (int i = 0; i < inputLayerNum; i++) {
                xj[j] += xi[i] * inWeight[i][j];
            }
        }
        // 隐层使用S函数进行激活
        for (int j = 0; j < hiddenLayerNum; j++) {
            xjActive[j] = 1 / (1 + Math.exp(-xj[j]));
        }
        // 隐层到输出层权值和计算
        double[] result = new double[outputLayerNum];
        for (int k = 0; k < outputLayerNum; k++) {
            xk[k] = 0;
            for (int j = 0; j < hiddenLayerNum; j++) {
                xk[k] += xjActive[j] * outWeight[j][k];
                result[k] = xk[k];
            }
        }
        return result;
    }
    
    /**
     * 训练时要注意数据规约到[-1,1]或者[0,1]，否则实际输出会出现NaN或者结果之间非常相近
     */
    public void train() {
        System.out.println("开始训练...");
        Random random = new Random(System.currentTimeMillis());
        while (curError > error) {
            Map<double[], double[]> testSet = new HashMap<double[], double[]>();
            for (int i = 1; i <= 20; i++) {
                double[] input = new double[1];
                double[] output = new double[1];
                input[0] = (random.nextInt(100) + 1);
                output[0] = getOutput(input[0]) / 10000.0;
                input[0] = input[0] / 10000.0;
                testSet.put(input, output);
            }
            System.out.println("curError:" + curError + " error:" + error);
            Set<Entry<double[],double[]>> entrySet = testSet.entrySet();
            for (Entry<double[], double[]> entry : entrySet) {
                forwardProcess(entry.getKey(), entry.getValue());
                backProcess();
            }
        }
        System.out.println("训练结束！");
    }
    
    public static void main(String[] args) {
        int il = 1;
        int hl = 10;
        int ol = 1;
        
        // 模拟函数 y = x ^ x - x， 测试算法
        BPNet bpNet = new BPNet(il, hl, ol);
        bpNet.train();
        Random random = new Random(System.currentTimeMillis());
        for (int i = 0; i < 10; i++) {
            double[] input = new double[1];
            double[] output = new double[1];
            input[0] = (random.nextInt(100) + 1);
            output[0] = getOutput(input[0]);
            input[0] = input[0] / 10000.0;
            double[] result = bpNet.calculate(input);
            System.out.println("输入：" + (input[0] * 10000) + " 理想输出：" + (output[0]) 
                    + " 实际输出：" + (result[0] * 10000));
        }
    }

    /**
     * 使用BP模拟函数 y = x ^ x - x
     * @param input
     * @return
     */
    private static double getOutput(double input) {
        return input * input - input;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
}