import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '时间序列预测',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        时间序列预测是一种统计技术，用于预测未来事件或趋势，基于历史数据的时间序列。
        它涉及分析按时间顺序排列的数据点，以识别模式、趋势和季节性变化，然后利用这
        些信息来预测未来的数据点。
      </>
    ),
  },
  {
    title: '图像类',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        图像类任务是指在计算机视觉领域中，涉及到图像数据的处理、分析和理解的各种任务。
        这些任务通常需要算法能够从图像中提取信息、识别对象、理解场景或执行其他视觉相关
        的功能。
      </>
    ),
  },
  {
    title: '其他行业',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
       to be determined
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
