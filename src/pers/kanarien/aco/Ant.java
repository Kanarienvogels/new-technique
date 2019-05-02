package pers.kanarien.aco;

import java.util.Random;
import java.util.ArrayList;

/**
 * 蚂蚁类，主要包含两个操作：初始化和选择下一个城市
 * @author Kanarien 
 * @version 1.0
 * @date 2019年4月2日 下午8:52:54
 */
public class Ant {

    private ArrayList<Integer> tabu; // 禁忌表，记录走过的城市
    private ArrayList<Integer> allowedCities; // 允许搜索的城市
    private float[][] delta; // 信息素变化矩阵 
    private int[][] distance; // 距离矩阵
    private float alpha; // α参数
    private float beta; // β参数
 
    private int tourLength; // 路径长度
    private int cityNum; // 城市数量
    private int firstCity; // 起始城市
    private int currentCity; // 当前城市
 
    /**
     * @param num 城市数量
     */
    public Ant(int num) {
        cityNum = num;
    }
 
    /**
     * 初始化蚂蚁，随机选择起始位置
     * @param distance 距离矩阵
     * @param a alpha
     * @param b beta
     */
    public void init(int[][] distance, float a, float b) {
        alpha = a;
        beta = b;
        // 初始允许搜索的城市集合
        allowedCities = new ArrayList<Integer>();
        // 初始禁忌表
        tabu = new ArrayList<Integer>();
        // 初始距离矩阵
        this.distance = distance;
        // 初始信息素变化矩阵为0，同时添加允许城市
        delta = new float[cityNum][cityNum];
        for (int i = 0; i < cityNum; i++) {
            Integer integer = new Integer(i);
            allowedCities.add(integer);
            for (int j = 0; j < cityNum; j++) {
                delta[i][j] = 0.f;
            }
        }
        // 随机挑选一个城市作为起始城市
        Random random = new Random(System.currentTimeMillis());
        firstCity = random.nextInt(cityNum);
        // 允许搜索的城市集合中移除起始城市
        for (Integer i : allowedCities) {
            if (i.intValue() == firstCity) {
                allowedCities.remove(i);
                break;
            }
        }
        // 将起始城市添加至禁忌表
        tabu.add(Integer.valueOf(firstCity));
        // 当前城市为起始城市
        currentCity = firstCity;
    }
 
    /**
     * 选择下一个城市
     * @param pheromone 信息素矩阵
     */
    public void selectNextCity(float[][] pheromone) {
        float[] p = new float[cityNum];
        float sum = 0.0f;
        // 计算共同的分母
        for (Integer i : allowedCities) {
            sum += Math.pow(pheromone[currentCity][i.intValue()], alpha)
                    * Math.pow(1.0 / distance[currentCity][i.intValue()], beta);
        }
        // 计算概率矩阵，即分子
        for (int i = 0; i < cityNum; i++) {
            boolean flag = false;
            for (Integer j : allowedCities) {
                if (i == j.intValue()) {
                    p[i] = (float) (Math.pow(pheromone[currentCity][i], alpha) * Math
                            .pow(1.0 / distance[currentCity][i], beta)) / sum;
                    flag = true;
                    break;
                }
            }
            if (flag == false) {
                p[i] = 0.f;
            }
        }
        // 轮盘赌选择下一个城市
        Random random = new Random(System.currentTimeMillis());
        float selectP = random.nextFloat();
        int selectCity = 0;
        float sum1 = 0.f;
        for (int i = 0; i < cityNum; i++) {
            sum1 += p[i];
            if (sum1 >= selectP) {
                selectCity = i;
                break;
            }
        }
        // 从允许选择的城市中去除select city
        for (Integer i : allowedCities) {
            if (i.intValue() == selectCity) {
                allowedCities.remove(i);
                break;
            }
        }
        // 在禁忌表中添加select city
        tabu.add(Integer.valueOf(selectCity));
        // 将当前城市改为选择的城市
        currentCity = selectCity;
    }
 
    /**
     * 计算路径长度
     * @return 路径长度
     */
    private int calculateTourLength() {
        int len = 0;
        //禁忌表tabu最终形式：起始城市,城市1,城市2...城市n,起始城市
        for (int i = 0; i < cityNum; i++) {
            len += distance[this.tabu.get(i).intValue()][this.tabu.get(i + 1).intValue()];
        }
        return len;
    }
 
    public ArrayList<Integer> getAllowedCities() {
        return allowedCities;
    }
 
    public void setAllowedCities(ArrayList<Integer> allowedCities) {
        this.allowedCities = allowedCities;
    }
 
    public int getTourLength() {
        tourLength = calculateTourLength();
        return tourLength;
    }
 
    public void setTourLength(int tourLength) {
        this.tourLength = tourLength;
    }
 
    public int getCityNum() {
        return cityNum;
    }
 
    public void setCityNum(int cityNum) {
        this.cityNum = cityNum;
    }
 
    public ArrayList<Integer> getTabu() {
        return tabu;
    }
 
    public void setTabu(ArrayList<Integer> tabu) {
        this.tabu = tabu;
    }
 
    public float[][] getDelta() {
        return delta;
    }
 
    public void setDelta(float[][] delta) {
        this.delta = delta;
    }
 
    public int getFirstCity() {
        return firstCity;
    }
 
    public void setFirstCity(int firstCity) {
        this.firstCity = firstCity;
    }

}
