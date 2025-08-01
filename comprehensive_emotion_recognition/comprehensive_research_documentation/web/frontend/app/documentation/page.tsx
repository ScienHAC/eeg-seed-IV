'use client';

import React from 'react';

export default function Documentation() {
  const documentationFiles = [
    {
      title: 'Comprehensive EEG Emotion Research Blueprint',
      filename: 'COMPREHENSIVE_EEG_EMOTION_RESEARCH_BLUEPRINT.md',
      description: 'Complete research paper draft with methodology, results, and analysis',
      icon: '📄',
      sections: ['Abstract', 'Introduction', 'Methodology', 'Results', 'Discussion', 'Conclusion']
    },
    {
      title: 'Active Files and Folders Reference',
      filename: 'ACTIVE_FILES_AND_FOLDERS_REFERENCE.txt',
      description: 'Complete guide to project structure and file usage',
      icon: '📁',
      sections: ['Core Folders', 'Working Models', 'Data Structure', 'Usage Guide']
    },
    {
      title: 'Future Stages Detailed Plan',
      filename: 'FUTURE_STAGES_DETAILED_PLAN.md',
      description: 'Roadmap for Stages 3-6 development with implementation details',
      icon: '🚀',
      sections: ['Stage 3: Autoencoders', 'Stage 4: CNNs', 'Stage 5: LSTMs', 'Stage 6: Ensemble']
    },
    {
      title: 'Technical Algorithms Reference',
      filename: 'TECHNICAL_ALGORITHMS_REFERENCE.md',
      description: 'Mathematical foundations and algorithm implementations',
      icon: '🔬',
      sections: ['Feature Engineering', 'ML Algorithms', 'Evaluation Metrics', 'Code Examples']
    },
    {
      title: 'Comprehensive Research Paper Draft',
      filename: 'COMPREHENSIVE_EEG_RESEARCH_PAPER_DRAFT.md',
      description: 'Publication-ready research paper with complete analysis',
      icon: '📝',
      sections: ['Literature Review', 'Dataset Analysis', 'Six-Stage Architecture', 'Results & Discussion']
    }
  ];

  const quickLinks = [
    {
      title: 'Getting Started',
      links: [
        { name: 'Quick Start Guide', href: '#quick-start' },
        { name: 'Installation', href: '#installation' },
        { name: 'Dataset Setup', href: '#dataset-setup' },
        { name: 'Running Models', href: '#running-models' }
      ]
    },
    {
      title: 'Research System',
      links: [
        { name: 'Project Structure', href: '#structure' },
        { name: 'Model Architecture', href: '#architecture' },
        { name: 'Feature Engineering', href: '#features' },
        { name: 'Results Analysis', href: '#results' }
      ]
    },
    {
      title: 'Development',
      links: [
        { name: 'Contributing', href: '#contributing' },
        { name: 'Future Stages', href: '#future' },
        { name: 'Code Standards', href: '#standards' },
        { name: 'Testing', href: '#testing' }
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-4">
              📚 Research Documentation
            </h1>
            <p className="text-xl text-indigo-100 mb-6">
              Comprehensive documentation for the 97.7% accuracy EEG emotion recognition system
            </p>
            <div className="bg-white/10 rounded-lg px-6 py-3 inline-block">
              <span className="text-sm">
                Complete research blueprints, technical references, and implementation guides
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Quick Navigation */}
        <div className="grid md:grid-cols-3 gap-8 mb-12">
          {quickLinks.map((section, index) => (
            <div key={index} className="research-card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">{section.title}</h3>
              <ul className="space-y-2">
                {section.links.map((link, linkIndex) => (
                  <li key={linkIndex}>
                    <a
                      href={link.href}
                      className="text-blue-600 hover:text-blue-800 text-sm transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Documentation Files */}
        <div className="research-card p-8 mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            📄 Research Documentation Suite
          </h2>
          <div className="grid gap-6">
            {documentationFiles.map((doc, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">{doc.icon}</div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{doc.title}</h3>
                      <p className="text-sm text-gray-600 mt-1">{doc.description}</p>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <button className="btn-secondary text-xs py-1 px-3">
                      View
                    </button>
                    <button className="btn-primary text-xs py-1 px-3">
                      Download
                    </button>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-2 mt-4">
                  {doc.sections.map((section, sectionIndex) => (
                    <span
                      key={sectionIndex}
                      className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-xs"
                    >
                      {section}
                    </span>
                  ))}
                </div>
                
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <span className="text-xs font-mono text-gray-500">{doc.filename}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Start Guide */}
        <div className="research-card p-8 mb-12" id="quick-start">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🚀 Quick Start Guide
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">1. System Requirements</h3>
              <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
                <div className="space-y-1">
                  <div>Python 3.8+</div>
                  <div>NumPy, Pandas, Scikit-learn</div>
                  <div>Matplotlib, Seaborn</div>
                  <div>Scipy (for .mat file handling)</div>
                  <div>16GB+ RAM recommended</div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">2. Installation</h3>
              <div className="bg-gray-900 text-gray-100 rounded-lg p-4 font-mono text-sm">
                <div className="space-y-1">
                  <div># Clone the repository</div>
                  <div className="text-green-400">git clone [repository-url]</div>
                  <div></div>
                  <div># Navigate to main system</div>
                  <div className="text-green-400">cd comprehensive_emotion_recognition</div>
                  <div></div>
                  <div># Install dependencies</div>
                  <div className="text-green-400">pip install -r requirements.txt</div>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-8">
            <h3 className="font-semibold text-gray-900 mb-4">3. Running the System</h3>
            <div className="bg-gray-900 text-gray-100 rounded-lg p-4 font-mono text-sm">
              <div className="space-y-2">
                <div># Execute the complete pipeline</div>
                <div className="text-green-400">python main.py</div>
                <div></div>
                <div># Check results</div>
                <div className="text-green-400">cat csv_data/comprehensive_report.txt</div>
                <div></div>
                <div># Expected output: 97.7% accuracy achievement</div>
              </div>
            </div>
          </div>
        </div>

        {/* System Architecture */}
        <div className="research-card p-8 mb-12" id="architecture">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🏗️ System Architecture
          </h2>
          
          <div className="space-y-6">
            <div className="bg-blue-50 rounded-lg p-6">
              <h3 className="font-semibold text-blue-900 mb-3">Main Research System</h3>
              <div className="font-mono text-sm text-blue-800">
                comprehensive_emotion_recognition/
                <br />├── 📄 main.py (Entry point - 97.7% system)
                <br />├── 📄 config.py (Six-stage configuration)
                <br />├── 📁 models/ (Stage 1 & 2 implementations)
                <br />├── 📁 data_processing/ (Complete pipeline)
                <br />├── 📁 csv_data/ (Results and reports)
                <br />└── 📁 comprehensive_research_documentation/
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-lg p-4">
                <h4 className="font-medium text-green-900 mb-2">✅ Completed Stages</h4>
                <ul className="text-sm text-green-800 space-y-1">
                  <li>• Stage 1: SVM Baseline (77.64%)</li>
                  <li>• Stage 2: Random Forest + SFS (97.7%)</li>
                  <li>• Complete data processing pipeline</li>
                  <li>• Feature engineering framework</li>
                </ul>
              </div>
              
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-blue-900 mb-2">🚀 Planned Stages</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Stage 3: Autoencoder features</li>
                  <li>• Stage 4: CNN spatial modeling</li>
                  <li>• Stage 5: LSTM temporal modeling</li>
                  <li>• Stage 6: Advanced ensemble</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Research Impact */}
        <div className="research-card p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            🎯 Research Impact & Applications
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Academic Contributions</h3>
              <ul className="space-y-3">
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">97.7% accuracy benchmark on SEED-IV dataset</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Comprehensive six-stage development framework</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Advanced feature engineering methodology</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Open-source reproducible research system</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Practical Applications</h3>
              <ul className="space-y-3">
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Mental health monitoring systems</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Brain-computer interfaces</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Emotion-aware human-computer interaction</span>
                </li>
                <li className="flex items-start space-x-3">
                  <div className="w-2 h-2 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-gray-700">Neurofeedback therapy systems</span>
                </li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 p-4 bg-indigo-50 rounded-lg">
            <p className="text-sm text-indigo-800">
              <strong>Research Status:</strong> This system represents a breakthrough in EEG-based emotion recognition, 
              with complete documentation enabling full reproducibility and future development. 
              All code, data processing scripts, and research documentation are available for academic and research use.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
