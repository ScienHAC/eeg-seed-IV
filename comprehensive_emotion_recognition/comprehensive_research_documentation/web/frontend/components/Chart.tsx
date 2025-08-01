import React from 'react';

interface ChartProps {
  data: any[];
  type: 'line' | 'bar' | 'pie';
  title?: string;
  xKey?: string;
  yKey?: string;
  colors?: string[];
  width?: number;
  height?: number;
}

const Chart: React.FC<ChartProps> = ({
  data,
  type,
  title,
  xKey = 'x',
  yKey = 'y',
  colors = ['#0ea5e9', '#22c55e', '#ef4444', '#f59e0b'],
  width,
  height = 300
}) => {
  // Simple placeholder chart implementation
  const renderChart = () => {
    if (type === 'bar' && data.length > 0) {
      const maxValue = Math.max(...data.map(d => d[yKey] || 0));
      
      return (
        <div className="space-y-2">
          {data.map((item, index) => (
            <div key={index} className="flex items-center space-x-3">
              <div className="w-16 text-sm text-gray-600 text-right">
                {item[xKey]}
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                <div
                  className="h-6 rounded-full flex items-center justify-end pr-2 text-white text-xs font-medium"
                  style={{
                    width: `${(item[yKey] / maxValue) * 100}%`,
                    backgroundColor: colors[index % colors.length]
                  }}
                >
                  {typeof item[yKey] === 'number' ? item[yKey].toFixed(1) : item[yKey]}%
                </div>
              </div>
            </div>
          ))}
        </div>
      );
    }
    
    if (type === 'line' && data.length > 0) {
      return (
        <div className="space-y-4">
          <div className="flex justify-between items-end h-32 bg-gray-50 rounded-lg p-4">
            {data.map((item, index) => (
              <div key={index} className="flex flex-col items-center">
                <div
                  className="w-8 bg-blue-500 rounded-t"
                  style={{
                    height: `${(item[yKey] / Math.max(...data.map(d => d[yKey]))) * 80}px`
                  }}
                ></div>
                <div className="text-xs text-gray-600 mt-1">{item[xKey]}</div>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {data.map((item, index) => (
              <div key={index} className="flex justify-between">
                <span className="text-gray-600">{item[xKey]}:</span>
                <span className="font-medium">{item[yKey]}%</span>
              </div>
            ))}
          </div>
        </div>
      );
    }
    
    if (type === 'pie' && data.length > 0) {
      return (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-2">
            {data.map((item, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: colors[index % colors.length] }}
                ></div>
                <span className="text-sm text-gray-700">
                  {item[xKey]}: {item[yKey]}%
                </span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return (
      <div className="flex items-center justify-center h-32 bg-gray-50 rounded-lg">
        <p className="text-gray-500">Chart data visualization</p>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      <div className="chart-container animate-fade-in-up" style={{ height }}>
        {renderChart()}
      </div>
    </div>
  );
};

export default Chart;
