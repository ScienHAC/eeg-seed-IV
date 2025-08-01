import React from 'react';

interface PerformanceTableProps {
  data: Array<{
    emotion: string;
    precision: number;
    recall: number;
    f1Score: number;
    support: number;
    color?: string;
  }>;
  title?: string;
}

const PerformanceTable: React.FC<PerformanceTableProps> = ({ data, title }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-6">
      {title && (
        <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      )}
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Emotion
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Precision
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Recall
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                F1-Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Support
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {row.color && (
                      <div 
                        className="w-3 h-3 rounded-full mr-3"
                        style={{ backgroundColor: row.color }}
                      />
                    )}
                    <span className="text-sm font-medium text-gray-900">
                      {row.emotion}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {(row.precision * 100).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {(row.recall * 100).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {(row.f1Score * 100).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {row.support}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Summary Row */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex justify-between items-center text-sm">
          <span className="font-semibold text-gray-900">
            Macro Average
          </span>
          <div className="flex space-x-8">
            <span className="text-gray-600">
              Precision: {((data.reduce((sum, row) => sum + row.precision, 0) / data.length) * 100).toFixed(1)}%
            </span>
            <span className="text-gray-600">
              Recall: {((data.reduce((sum, row) => sum + row.recall, 0) / data.length) * 100).toFixed(1)}%
            </span>
            <span className="text-gray-600">
              F1-Score: {((data.reduce((sum, row) => sum + row.f1Score, 0) / data.length) * 100).toFixed(1)}%
            </span>
            <span className="text-gray-600">
              Total: {data.reduce((sum, row) => sum + row.support, 0)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PerformanceTable;
