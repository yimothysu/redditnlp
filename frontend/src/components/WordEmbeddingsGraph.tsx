import { Scatter } from 'react-chartjs-2';

import {  Chart,  PointElement,  LinearScale, } from 'chart.js';

import annotationPlugin from 'chartjs-plugin-annotation';

Chart.register(PointElement, LinearScale, annotationPlugin);

function WordEmbeddingsGraph({ embeddings, isComparisonMode }: { embeddings: { word: string; x: number; y: number }[], isComparisonMode: string }) {
       const annotations: { [param: number]: any } = {};

       embeddings.map(({ x, y, word }, index) => {
        annotations[index] = {
            type: 'line',
            borderWidth: 3,
            xMin: 0, yMin: 0,
            xMax: x, yMax: y,
            arrowHeads: { end: { display: true } } 
        };    
        
        annotations[index + embeddings.length] = {
            type: 'label', 
            xValue: x / 2, yValue: y / 2,
            content: [word], color: 'black',
            padding: 10,
            textAlign: 'center',
            font: { size: 12, weight: '600'},
            backgroundColor: 'rgba(255, 255, 255, 1)'
        }; });
        return (
        <div style={{ width: isComparisonMode === 'true' ? '40vw' :'90vw',  height: 'auto'}}>
        <Scatter data={{
            datasets: [{
                data: embeddings, pointRadius: 3
            }
        ] }}
        options={{
            layout: {
                padding: 10
            },

            scales: { 
                y: { min: -4, max: 4 }, x: { min: -4, max: 4 }
            }, 
            plugins: {
                annotation: { annotations },
                legend: { display: false }
            },
            //maintainAspectRatio: false
        }}
    /></div>); 
}
    
export default WordEmbeddingsGraph;