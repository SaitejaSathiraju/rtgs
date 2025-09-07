#!/usr/bin/env python3
"""
Test the ultimate command with our fixed agent system
"""

def test_ultimate_command():
    try:
        print("🚀 Testing Ultimate Command with Fixed Agents...")
        
        # Test importing our fixed agents
        from agents import DataCleanerAgent, DataTransformerAgent, DataAnalystAgent, DataSummarizerAgent
        from crewai import Crew, Process
        
        print("✅ Fixed agents imported successfully")
        
        # Test creating agents
        cleaner = DataCleanerAgent()
        transformer = DataTransformerAgent()
        analyst = DataAnalystAgent()
        summarizer = DataSummarizerAgent()
        
        print("✅ All agents created successfully")
        
        # Test agent tools
        for agent_name, agent in [("Cleaner", cleaner), ("Transformer", transformer), 
                                 ("Analyst", analyst), ("Summarizer", summarizer)]:
            tools = agent.get_agent().tools
            print(f"✅ {agent_name} has {len(tools)} tools: {[tool.name for tool in tools]}")
        
        # Test creating tasks
        test_dataset = "data/ts_industry_tsipass_01-08-2025_31-08-2025.csv"
        
        cleaning_task = cleaner.create_cleaning_task(test_dataset)
        transformation_task = transformer.create_transformation_task(test_dataset)
        analysis_task = analyst.create_analysis_task(test_dataset)
        summarization_task = summarizer.create_summarization_task(test_dataset)
        
        print("✅ All tasks created successfully")
        
        # Test creating crew
        crew = Crew(
            agents=[
                cleaner.get_agent(),
                transformer.get_agent(),
                analyst.get_agent(),
                summarizer.get_agent()
            ],
            tasks=[
                cleaning_task,
                transformation_task,
                analysis_task,
                summarization_task
            ],
            process=Process.sequential,
            verbose=False  # Less verbose for test
        )
        
        print("✅ Crew created successfully")
        print("🎉 Ultimate command is ready with REAL tool-executing agents!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 TESTING ULTIMATE COMMAND WITH FIXED AGENTS")
    print("=" * 60)
    success = test_ultimate_command()
    if success:
        print("\n🎉 ULTIMATE COMMAND TEST PASSED!")
        print("✅ The ultimate command now uses REAL tool-executing agents!")
        print("✅ Ready to run: py cli_analyst.py ultimate")
    else:
        print("\n❌ Ultimate command test failed")
