package pers.kanarien.aco;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class AntSystem {

    private Ant[] ants; // 蚂蚁
    private int antNum; // 蚂蚁数量
    private int cityNum; // 城市数量
    private int maxIteration; // 迭代数
    private float[][] pheromone; // 信息素矩阵
    private int[][] distance; // 距离矩阵
    private int bestLength; // 最佳长度
    private int[] bestTour; // 最佳路径
 
    // 三个参数
    private float alpha;
    private float beta;
    private float rho;
 
    /**
     * @param n 城市数量
     * @param m 蚂蚁数量
     * @param g 运行代数
     * @param a alpha
     * @param b beta
     * @param r rho
     * 
     **/
    public AntSystem(int n, int m, int g, float a, float b, float r) {
        cityNum = n;
        antNum = m;
        ants = new Ant[antNum];
        maxIteration = g;
        alpha = a;
        beta = b;
        rho = r;
    }
 
    /**
     * 初始化蚂蚁系统
     * @param filename 数据文件名，该文件存储所有城市节点坐标数据
     * @throws IOException
     */
    @SuppressWarnings("resource")
    private void init(String filename) throws IOException {
        // 读取数据
        int[] x;
        int[] y;
        String strbuff;
        BufferedReader data = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        distance = new int[cityNum][cityNum];
        x = new int[cityNum];
        y = new int[cityNum];
        for (int i = 0; i < cityNum; i++) {
            // 读取一行数据，数据格式为:城市序号 x轴坐标 y轴坐标，如1 6734 1453
            strbuff = data.readLine();
            String[] strcol = strbuff.split(" ");
            x[i] = Integer.valueOf(strcol[1]);// x坐标
            y[i] = Integer.valueOf(strcol[2]);// y坐标
        }
        // 计算距离矩阵
        // 距离取欧氏距离
        for (int i = 0; i < cityNum - 1; i++) {
            distance[i][i] = 0; // 对角线为0
            for (int j = i + 1; j < cityNum; j++) {
                double rij = Math.sqrt(((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j])));
                // 四舍五入，取整
                int tij = (int) Math.round(rij);
                distance[i][j] = tij;
                distance[j][i] = distance[i][j];
            }
        }
        distance[cityNum - 1][cityNum - 1] = 0;
        // 初始化信息素矩阵
        pheromone = new float[cityNum][cityNum];
        for (int i = 0; i < cityNum; i++) {
            for (int j = 0; j < cityNum; j++) {
                pheromone[i][j] = 0.1f; // 初始化为0.1
            }
        }
        bestLength = Integer.MAX_VALUE;
        bestTour = new int[cityNum + 1];
        // 随机放置蚂蚁
        for (int i = 0; i < antNum; i++) {
            ants[i] = new Ant(cityNum);
            ants[i].init(distance, alpha, beta);
        }
    }
 
    /**
     * 运行蚂蚁系统算法
     */
    public void solve() {
        System.out.println("开始运行蚂蚁系统算法（Ant System）...");
        System.out.println("蚂蚁数：" + antNum + " 迭代数：" + maxIteration + " α：" + alpha 
                + " β：" + beta + " ρ：" + rho);
        long beginTime = System.currentTimeMillis();
        // 迭代maxIteration次
        for (int g = 0; g < maxIteration; g++) {
            // antNum只蚂蚁
            for (int i = 0; i < antNum; i++) {
                // i这只蚂蚁走cityNum步，完整一个TSP
                for (int j = 1; j < cityNum; j++) {
                    ants[i].selectNextCity(pheromone);
                }
                // 把这只蚂蚁起始城市加入其禁忌表中
                // 禁忌表最终形式：起始城市,城市1,城市2...城市n,起始城市
                ants[i].getTabu().add(ants[i].getFirstCity());
                // 查看这只蚂蚁行走路径距离是否比当前距离优秀
                int tourLength = ants[i].getTourLength();
                if (tourLength < bestLength) {
                    // 比当前优秀则拷贝优秀TSP路径
                    bestLength = tourLength;
                    for (int k = 0; k < cityNum + 1; k++) {
                        bestTour[k] = ants[i].getTabu().get(k).intValue();
                    }
                }
                // 更新这只蚂蚁的信息数变化矩阵，对称矩阵
                for (int j = 0; j < cityNum; j++) {
                    ants[i].getDelta()[ants[i].getTabu().get(j).intValue()][ants[i]
                            .getTabu().get(j + 1).intValue()] = (float) (1.0 / tourLength);
                    ants[i].getDelta()[ants[i].getTabu().get(j + 1).intValue()][ants[i]
                            .getTabu().get(j).intValue()] = (float) (1.0 / tourLength);
                }
            }
            // 更新信息素
            updatePheromone();
            // 重新初始化蚂蚁
            for (int i = 0; i < antNum; i++) {
                ants[i].init(distance, alpha, beta);
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("算法结束，耗时 " + (endTime - beginTime) + "ms");
        // 迭代完后，打印最佳结果
        printOptimal();
    }
 
    /**
     * 更新信息素
     */
    private void updatePheromone() {
        // 信息素挥发
        for (int i = 0; i < cityNum; i++)
            for (int j = 0; j < cityNum; j++)
                pheromone[i][j] = pheromone[i][j] * (1 - rho);
        // 信息素更新
        for (int i = 0; i < cityNum; i++) {
            for (int j = 0; j < cityNum; j++) {
                for (int k = 0; k < antNum; k++) {
                    pheromone[i][j] += ants[k].getDelta()[i][j];
                }
            }
        }
    }
 
    private void printOptimal() {
        System.out.println("最优长度为: " + bestLength);
        System.out.println("最优路径为: ");
        for (int i = 0; i < cityNum + 1; i++) {
            System.out.format("%3d", bestTour[i]);
            if ((i + 1) % 20 == 0) {
                System.out.println();
            }
        }
        System.out.println();
    }
 
 
    /**
     * 算法入口
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        int cityNum = 52; // 城市数量
        int antNum = 50; // 蚂蚁数量
        int maxIteration = 200; // 迭代次数
        float alpha = 1.0f; // α参数
        float beta = 3.0f; // β参数
        float rho = 0.5f; // ρ参数
        String filename = "data/berlin52.txt";
        AntSystem aco = new AntSystem(cityNum, antNum, maxIteration, alpha, beta, rho);
        aco.init(filename);
        aco.solve();
    }

}
