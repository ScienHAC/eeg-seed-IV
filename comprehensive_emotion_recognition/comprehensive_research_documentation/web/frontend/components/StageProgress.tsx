import React from 'react';

interface StageProgressProps {
  currentStage: number;
  totalStages: number;
  stageNames: string[];
  completedStages: number[];
}

const StageProgress: React.FC<StageProgressProps> = ({ 
  currentStage, 
  totalStages, 
  stageNames, 
  completedStages 
}) => {
  return (
    <div className="w-full bg-white rounded-lg p-6 shadow-lg border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Six-Stage Development Progress
      </h3>
      
      <div className="space-y-4">
        {Array.from({ length: totalStages }, (_, index) => {
          const stageNum = index + 1;
          const isCompleted = completedStages.includes(stageNum);
          const isCurrent = stageNum === currentStage;
          const isPlanned = stageNum > Math.max(...completedStages);
          
          return (
            <div key={stageNum} className="flex items-center space-x-4">
              {/* Stage Number Circle */}
              <div
                className={`flex items-center justify-center w-10 h-10 rounded-full text-sm font-semibold ${
                  isCompleted
                    ? 'bg-green-500 text-white'
                    : isCurrent
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-500'
                }`}
              >
                {isCompleted ? '✓' : stageNum}
              </div>
              
              {/* Stage Info */}
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span
                    className={`font-medium ${
                      isCompleted
                        ? 'text-green-700'
                        : isCurrent
                        ? 'text-blue-700'
                        : 'text-gray-500'
                    }`}
                  >
                    Stage {stageNum}: {stageNames[index] || `Stage ${stageNum}`}
                  </span>
                  
                  <span
                    className={`text-xs px-2 py-1 rounded-full ${
                      isCompleted
                        ? 'bg-green-100 text-green-700'
                        : isCurrent
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-gray-100 text-gray-500'
                    }`}
                  >
                    {isCompleted ? 'Completed' : isPlanned ? 'Planned' : 'In Progress'}
                  </span>
                </div>
                
                {/* Progress Bar */}
                <div className="mt-2 stage-progress-bar">
                  <div
                    className={`stage-progress-fill ${
                      isCompleted ? 'w-full' : isCurrent ? 'w-3/4' : 'w-0'
                    }`}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Summary Stats */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-green-600">
              {completedStages.length}
            </div>
            <div className="text-sm text-gray-500">Completed</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {currentStage && !completedStages.includes(currentStage) ? 1 : 0}
            </div>
            <div className="text-sm text-gray-500">In Progress</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-400">
              {totalStages - completedStages.length - (currentStage && !completedStages.includes(currentStage) ? 1 : 0)}
            </div>
            <div className="text-sm text-gray-500">Planned</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StageProgress;
