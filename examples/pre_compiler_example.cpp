//
// Created by gyankos on 07/10/26.
//

#include <torch/torch.h>
#include <logds/libtorchwrapper/general_training_framework.h>

std::shared_ptr<torch::jit::Graph> create_user_loss_and_stopping_subgraph() {
    auto u_graph = std::make_shared<torch::jit::Graph>();
    // Creazione del TupleType puro (senza nome o label di chiavi).
    // Questo è il formato nativo che bypassa l'asserzione di "convertNamedType" nel serializer.
    // 1. Establish the explicit named structure for your return types
    auto tensor_type = c10::TensorType::get();
    auto bool_type = c10::BoolType::get();
    auto pure_tuple_type = c10::TupleType::create({tensor_type, bool_type});

    // 1. Define the abstract JIT Types we need
    // auto tensor_type = c10::TensorType::get();               // The Type representing a Tensor
    auto tensor_list_type = c10::ListType::create(tensor_type); // The Type representing List[Tensor]

    // 2. Add inputs (LibTorch defaults them all to TensorType)
    torch::jit::Value* u_weights = u_graph->addInput("weights");
    torch::jit::Value* u_inputs_list = u_graph->addInput("inputs_list");
    torch::jit::Value* u_targets_list = u_graph->addInput("targets_list");
    torch::jit::Value* u_loop_iter = u_graph->addInput("loop_iter");
    torch::jit::Value* u_loop_cond_in = u_graph->addInput("loop_cond_in");

    // 3. CRITICAL: Override the default Tensor types with correct static list types
    u_inputs_list->setType(tensor_list_type);
    u_targets_list->setType(tensor_list_type);

    // If 'weights' is a List[Tensor] as well, set its type:
    u_weights->setType(tensor_list_type);
    // If 'weights' is just a big 2D Tensor, keep it as is, but change line 33
    // from aten::__getitem__ to aten::select or an explicit tensor slice.

    // 4. Fix loop_iter type if it represents a counter scalar
    u_loop_iter->setType(c10::IntType::get());
    u_loop_cond_in->setType(c10::BoolType::get());

    // --- Rest of your code remains identical ---
    torch::jit::Value* alpha_one = u_graph->insertConstant(1.0);
    torch::jit::Value* none_val = u_graph->insertConstant(c10::nullopt);

    torch::jit::Value* num_batches = u_graph->insertConstant(2);
    torch::jit::Node* batch_idx = u_graph->create(torch::jit::aten::remainder, {u_loop_iter, num_batches});
    u_graph->insertNode(batch_idx);

    c10::Symbol get_item_op = c10::Symbol::fromQualString("aten::__getitem__");
    torch::jit::Node* input_batch = u_graph->create(get_item_op, {u_inputs_list, batch_idx->output()});
    torch::jit::Node* target_batch = u_graph->create(get_item_op, {u_targets_list, batch_idx->output()});
    u_graph->insertNode(input_batch);
    u_graph->insertNode(target_batch);

    // Explicitly type the outputs of getitem so subsequent nodes (like matmul) pass validation
    input_batch->output()->setType(tensor_type);
    target_batch->output()->setType(tensor_type);

    torch::jit::Node* w0 = u_graph->create(get_item_op, {u_weights, u_graph->insertConstant(0)});
    u_graph->insertNode(w0);
    w0->output()->setType(tensor_type);

    torch::jit::Node* pred = u_graph->create(torch::jit::aten::matmul, {input_batch->output(), w0->output()});
    u_graph->insertNode(pred);
    pred->output()->setType(tensor_type);

    torch::jit::Node* sub = u_graph->create(torch::jit::aten::sub, {pred->output(), target_batch->output(), alpha_one});
    u_graph->insertNode(sub);
    sub->output()->setType(tensor_type);

    torch::jit::Node* pow_n = u_graph->create(torch::jit::aten::pow, {sub->output(), u_graph->insertConstant(2.0)});
    u_graph->insertNode(pow_n);
    pow_n->output()->setType(tensor_type);

    torch::jit::Node* loss = u_graph->create(torch::jit::aten::mean, {pow_n->output(), none_val});
    u_graph->insertNode(loss);
    loss->output()->setType(tensor_type);
    torch::jit::Value* current_loss = loss->output();

    torch::jit::Value* tolerance = u_graph->insertConstant(0.0001);
    torch::jit::Node* custom_check = u_graph->create(torch::jit::aten::gt, {current_loss, tolerance});
    u_graph->insertNode(custom_check);
    // custom_check->output()->setType(c10::BoolType::get());
    // torch::jit::Value* next_loop_cond = custom_check->output();
    // aten::gt restituisce un Tensor(bool), quindi assegniamo TensorType
    custom_check->output()->setType(tensor_type);

    // CRITICAL FIX: Convertiamo il Tensor di output in un Bool primitivo di TorchScript
    torch::jit::Node* bool_cast = u_graph->create(c10::Symbol::fromQualString("aten::Bool"), {custom_check->output()});
    u_graph->insertNode(bool_cast);
    bool_cast->output()->setType(c10::BoolType::get()); // Questo adesso è un vero bool!
    torch::jit::Value* next_loop_cond = bool_cast->output();

    createTupleResult(u_graph.get(), {current_loss, next_loop_cond}, {tensor_type, bool_type});

    return u_graph;
}


int main(void) {


    torch::jit::Module module_{"create_user_loss_and_stopping_subgraph"};
    auto graph = create_user_loss_and_stopping_subgraph();

    // 1. CRITICAL: Inject a "self" node at the absolute start of your input list (Index 0)
    // LibTorch methods require this so the serialization logic knows what module it belongs to.
    torch::jit::Value* self_val = graph->insertInput(0, "self");
    self_val->setType(module_._ivalue()->type()); // Bind it to this specific module instance type

    c10::QualifiedName method_name(*module_.type()->name(), "forward");

    // 2. Define the types exactly matching your updated graph configuration
    auto tensor_type = c10::TensorType::get();
    auto tensor_list_type = c10::ListType::create(tensor_type);
    auto int_type = c10::IntType::get();
    auto bool_type = c10::BoolType::get();

    // 3. Construct the argument list mirroring the graph inputs (including self)
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", module_.type()),
        c10::Argument("weights", tensor_list_type),
        c10::Argument("inputs_list", tensor_list_type),
        c10::Argument("targets_list", tensor_list_type),
        c10::Argument("loop_iter", int_type),
        c10::Argument("loop_cond_in", bool_type)
    };

    // 4. Construct the return signature tuple
    auto tuple_return_type = c10::TupleType::create({tensor_type, bool_type});
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    // 5. Build the formal schema
    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    // 6. Compile the graph using the valid two-parameter function signature
    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);

    // 7. Apply the schema directly to the compiled method
    method->setSchema(std::move(schema));

    // 8. Bind the configured method down to the module runtime layer
    module_.type()->addMethod(method);

    // This will now execute and output your serialized model file safely
    module_.save("create_user_loss_and_stopping_subgraph.pt");
}


int main_(void) {
     torch::jit::Module module("GeneralTrainingFramework");
    auto main_graph = std::make_shared<torch::jit::Graph>();

    torch::jit::Value* self = main_graph->addInput("self");
    self->setType(module._ivalue()->type());

    torch::jit::Value* in_dataset_inputs  = main_graph->addInput("dataset_inputs");
    torch::jit::Value* in_dataset_targets = main_graph->addInput("dataset_targets");
    torch::jit::Value* in_initial_weights = main_graph->addInput("initial_weights");

    auto tensor_list_type = c10::ListType::create(c10::TensorType::get());
    auto single_tensor_type = c10::TensorType::get();

    in_dataset_inputs->setType(tensor_list_type);
    in_dataset_targets->setType(tensor_list_type);
    in_initial_weights->setType(tensor_list_type);

    torch::jit::Node* w0_init = main_graph->create(torch::jit::aten::select, {in_initial_weights, main_graph->insertConstant(static_cast<int64_t>(0))});
    torch::jit::Node* w1_init = main_graph->create(torch::jit::aten::select, {in_initial_weights, main_graph->insertConstant(static_cast<int64_t>(1))});
    main_graph->insertNode(w0_init); main_graph->insertNode(w1_init);

    w0_init->output()->setType(single_tensor_type);
    w1_init->output()->setType(single_tensor_type);

    std::vector<torch::jit::Value*> weights_vector = {w0_init->output(), w1_init->output()};

    torch::jit::Node* in_b0 = main_graph->create(torch::jit::aten::select, {in_dataset_inputs, main_graph->insertConstant(static_cast<int64_t>(0))});
    torch::jit::Node* in_b1 = main_graph->create(torch::jit::aten::select, {in_dataset_inputs, main_graph->insertConstant(static_cast<int64_t>(1))});
    torch::jit::Node* tg_b0 = main_graph->create(torch::jit::aten::select, {in_dataset_targets, main_graph->insertConstant(static_cast<int64_t>(0))});
    torch::jit::Node* tg_b1 = main_graph->create(torch::jit::aten::select, {in_dataset_targets, main_graph->insertConstant(static_cast<int64_t>(1))});
    main_graph->insertNode(in_b0); main_graph->insertNode(in_b1); main_graph->insertNode(tg_b0); main_graph->insertNode(tg_b1);

    in_b0->output()->setType(single_tensor_type);
    in_b1->output()->setType(single_tensor_type);
    tg_b0->output()->setType(single_tensor_type);
    tg_b1->output()->setType(single_tensor_type);

    std::vector<torch::jit::Value*> inputs_vector = {in_b0->output(), in_b1->output()};
    std::vector<torch::jit::Value*> targets_vector = {tg_b0->output(), tg_b1->output()};

    // GENERAZIONE SOTTOGRAFI CONNESSI
    auto user_subgraph = create_user_loss_and_stopping_subgraph();
    auto opt_subgraph  = create_parametrizable_optimizer_subgraph(
        2, OptimizerType::Adam, DecayType::Exponential, 0.95f, 10
    );

    int max_epochs = 200;
    float learning_rate = 0.01f;

    // FUSIONE END-TO-END DELLA PIPELINE
    build_universal_training_pipeline(
        main_graph, inputs_vector, targets_vector, weights_vector,
        user_subgraph, opt_subgraph, max_epochs, learning_rate
    );

    // c10::QualifiedName method_name(*module.type()->name(), "optimize_problem");
    // auto cu = module._ivalue()->compilation_unit();
    // =========================================================================
    // REGISTRAZIONE DEL METODO CON SCHEMA TIPO-SICURO A SINGOLO RITORNO (LISTA)
    // =========================================================================
    c10::QualifiedName method_name(*module.type()->name(), "optimize_problem");
    auto cu = module._ivalue()->compilation_unit();

    std::vector<c10::Argument> arguments = {
        c10::Argument("self", module.type()),
        c10::Argument("dataset_inputs", tensor_list_type),
        c10::Argument("dataset_targets", tensor_list_type),
        c10::Argument("initial_weights", tensor_list_type)
    };

    // CORREZIONE: Il tipo di ritorno è direttamente la lista di Tensori (tensor_list_type), ovvero Tensor[]
    std::vector<c10::Argument> returns = {
        c10::Argument("", tensor_list_type)
    };

    // Creiamo la funzione legando lo schema tipato forte senza Tuple anonime
    auto schema = c10::FunctionSchema(method_name.name(), "", std::move(arguments), std::move(returns));
    auto compiled_function = cu->create_function(method_name, main_graph);
    compiled_function->setSchema(std::move(schema));

    // Aggiungiamo il metodo al tipo del modulo
    module.type()->addMethod(compiled_function);
    // std::vector<c10::Argument> arguments = {
    //     c10::Argument("self", module.type()),
    //     c10::Argument("dataset_inputs", tensor_list_type),
    //     c10::Argument("dataset_targets", tensor_list_type),
    //     c10::Argument("initial_weights", tensor_list_type)
    // };
    //
    // auto tuple_return_type = c10::TupleType::create({ tensor_list_type });
    // std::vector<c10::Argument> returns = { c10::Argument("", tuple_return_type) };
    //
    // auto schema = c10::FunctionSchema(method_name.name(), "", std::move(arguments), std::move(returns));
    // auto compiled_function = cu->create_function(method_name, main_graph);
    // compiled_function->setSchema(std::move(schema));
    //
    // module.type()->addMethod(compiled_function);
    //
    std::string filename = "universal_optimizer_module.pt";
    module.save(filename);
    // std::cout << "SUCCESS: Compilazione JIT completata. Modulo salvato in: " << filename << std::endl;

    return 0;
}
