// Domain Randomization Plugin for Gazebo
#include <gazebo/gazebo.hh>

namespace gazebo
{
    class DRPlugin(): public WorldPlugin()
    {
        public: DRPlugin() : WorldPlugin()
        {
            // needs to do something here
        }
        
        public: void DRPlugin::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
        {

        }
    }

    
}