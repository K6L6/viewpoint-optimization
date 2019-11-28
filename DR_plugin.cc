// Domain Randomization Plugin for Gazebo
#include <gazebo/gazebo.hh>

namespace gazebo
{
    class DRPlugin(): public WorldPlugin()
    {
        public: DRPlugin() : WorldPlugin()
        {
            gzmsg << "[DRPlugin] Launched." << std::endl;
        }
        
        DRPlugin::~DRPlugin()
        {
            gzmsg << "[DRPlugin] Unloaded plugin." << std::endl;
        }

        public: void DRPlugin::Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
        {
            NULL_CHECK(_world, "World pointer is NULL");
            this->world = _world
            this->physics_engine = world->Physics();

            loadTopicNames(_sdf); // read topic names from SDF

            this->update_connection = event::Events::ConnectWorldUpdateBegin(std::bind(&DRPlugin::onUpdate, this)); // connect to world update event

            // initialize transport node
            this->data_ptr->node = transport::NodePtr(new transport::Node());
            this->data_ptr->node->Init();

            this->data_ptr->sub = this->data_ptr->node->Subscribe(req_topic, &DRPlugin::onRequest, this); //subscribe to monitored request topic

            // Publish to the response topic
            this->data_ptr->pub = this->data_ptr->node->Advertise<DRResponse>(res_topic);
            gzmsg << "[DRPlugin] Loaded plugin." << std::endl;

        }

        void DRPlugin::processModel(const msgs::Model & msg)
        {
            physics::ModelPtr model;
            ignition::math::Vector3d scale;

            std::string model_name = msg.name();
            model = world->ModelByName(model_name);
            NULL_CHECK(model, "Model not found:" + model_name);

            for (const auto & joint : msg.joint())
            {
                processJoint(model, joint);
            }
            for (const auto & link : msg.link())
            {
                processLink(model, link);
            }
            if (msg.has_scale())
            {
                scale = msgs::ConvertIgn(msg.scale());
                model->SetScale(scale, true);
                gzdbg << "Scaled " << model_name << " by " << scale << std::endl;
            }
            for (const auto & nested_model : msg.model())
            {
                processModel(nested_model);
            }
        }

        void DRPlugin::processJoint(physics::ModelPtr model, const msgs::Joint & msg)
        {
            std::string joint_name;
            physics::JointPtr joint;
            msgs::Axis axis_msg;
            double value;

            NULL_CHECK(model, "Invalid model");
            joint_name = msg.name();
            joint = model->GetJoint(joint_name);
            NULL_CHECK(joint, "Joint not found: " + joint_name);

            gzdbg << "Processed joint " << joint_name << std::endl;
        }

        void DRPlugin::processLink(physics::ModelPtr model, const msgs::Link & msg)
        {
            std::string link_name;
            physics::LinkPtr link;
            msgs::Collision collision_msg;
            physics::CollisionPtr collision;

            link_name = msg.name();
            link = model->GetChildLink(link_name);
            NULL_CHECK(link, "Link not found");

        }

        void DRPlugin::processModelCmd(const ModelCmdMsg & msg)
        {
            std::string model_name;
            physics::ModelPtr model;

            model_name = msg.model_name();
            model = world->ModelByName(model_name);
            NULL_CHECK(model, "Model not found");

            for (const auto & joint_cmd : msg.joint_cmd())
            {
                processJointCmd(model, joint_cmd);
            }
        }

        void DRPlugin::processJointCmd(physics::ModelPtr model, const msgs::JointCmd & msg)
        {
            // Joint scoped name
            std::string joint_name;
            physics::JointPtr joint;
            physics::JointControllerPtr controller;

            NULL_CHECK(model, "Invalid model");
            controller = model->GetJointController();
            joint_name = msg.name();

            if (msg.has_position())
            {
                processPID(POSITION, controller, joint_name, msg.position());
            }
            if (msg.has_velocity())
            {
                processPID(VELOCITY, controller, joint_name, msg.velocity());
            }
        }

    }; // end of class bracket
    GZ_REGISTER_WORLD_PLUGIN(DRPlugin)
    
}