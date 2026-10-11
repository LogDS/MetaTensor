//
// Created by gyankos on 07/10/26.
//

#include <torch/torch.h>
#include <logds/libtorchwrapper/general_training_framework.h>
#include <torch/csrc/jit/ir/ir_views.h>

void printAttributes(std::ostream& out, torch::jit::Node* nodeptr, bool ignore_subgraph = false) {
    out << '[';
    auto names = nodeptr->attributeNames();
    int i = 0;
    for (auto name : names) {
        if (ignore_subgraph && name == c10::attr::Subgraph) {
            continue;
        }
        if (i++ > 0) {
            out << ", ";
        }
        // TODO: debugging mode to see the qualifier.  We definitely
        // don't want to print the qualifier since it should always
        // be attribute, but you might be able to track down a weird
        // bug by printing it out.
        out << name.toDisplayString() << ' ';
    }
    out << ']';
}

/**
 * An example of a subgraph which is providing the major computation force
 *
 * @return The compiled subgraph which is mimicking the loop iteration function.
 */
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
    auto tensor_list_type = c10::ListType::ofTensors(); // The Type representing List[Tensor]

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
    auto val = batch_idx->output();
    val->setType(c10::IntType::get());
    u_graph->insertNode(batch_idx);

    c10::Symbol get_item_op = c10::Symbol::fromQualString("aten::__getitem__");
    torch::jit::Node* input_batch = u_graph->create(get_item_op, {u_inputs_list, val});
    torch::jit::Node* target_batch = u_graph->create(get_item_op, {u_targets_list, val});
    // Explicitly type the outputs of getitem so subsequent nodes (like matmul) pass validation
    input_batch->output()->setType(tensor_type);
    target_batch->output()->setType(tensor_type);
    u_graph->insertNode(input_batch);
    u_graph->insertNode(target_batch);

    torch::jit::Node* w0 = u_graph->create(get_item_op, {u_weights, u_graph->insertConstant(0)->setType(c10::IntType::get())});
    w0->output()->setType(tensor_type);
    u_graph->insertNode(w0);


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

int driver_create_parametrizable_optimizer_subgraph() {
    torch::jit::Module module_{"create_parametrizable_optimizer_subgraph"};
    auto graph = create_parametrizable_optimizer_subgraph(3, OptimizerType::Adam, DecayType::Exponential, 0.03, 100);

    // 1. CRITICAL: Inject a "self" node at the absolute start of your input list (Index 0)
    // LibTorch methods require this so the serialization logic knows what module it belongs to.
    torch::jit::Value* self_val = graph->insertInput(0, "self");
    self_val->setType(module_._ivalue()->type()); // Bind it to this specific module instance type

    c10::QualifiedName method_name(*module_.type()->name(), "forward");

    auto tensor_type = c10::TensorType::get();
    auto tensor_list_type = c10::ListType::create(tensor_type);
    auto int_type = c10::IntType::get();
    auto bool_type = c10::BoolType::get();
    auto float_type = c10::FloatType::get();
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", module_.type()),
        c10::Argument("weights_in", tensor_list_type),
        c10::Argument("grads_in", tensor_list_type),
        c10::Argument("lr_in", float_type),
        c10::Argument("m_in", tensor_list_type),
        c10::Argument("v_in", tensor_list_type),
        c10::Argument("loop_iter", int_type)
    };

    // 4. Construct the return signature tuple
    auto tuple_return_type = c10::TupleType::create({tensor_list_type, float_type, tensor_list_type, tensor_list_type});
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);

    // 7. Apply the schema directly to the compiled method
    method->setSchema(std::move(schema));

    // 8. Bind the configured method down to the module runtime layer
    module_.type()->addMethod(method);

    // This will now execute and output your serialized model file safely
    module_.save("create_parametrizable_optimizer_subgraph.pt");
    return 0;
}

int driver_create_user_loss_and_stopping_subgraph() {
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
    return 0;
}

void print_block(torch::jit::Block * graphR, int indent = 0) {
    auto ind = std::string(indent, '\t');
    for (const auto& node : graphR->inputs()) {
        std::cout << ind <<  "in: " << node->debugName() << " : " << node->type()->repr_str() << std::endl;
    }
    for (const auto& node : graphR->outputs()) {
        std::cout << ind << "out: " << node->debugName() << " : " << node->type()->repr_str() << std::endl;
    }
    for (const auto& node : graphR->nodes()) {
        auto operator_ = std::string(node->kind().toDisplayString());
        if (node->kind() == c10::prim::PythonOp) {
            auto* pyOp = static_cast<const ::torch::jit::PythonOp*>(node);
            std::cout  << ind << '^' << pyOp->name() << std::endl;
        } else {
            std::cout << ind << node->kind().toQualString();
            if (node->hasAttributes()) {
                printAttributes(std::cout, node);
            }
            std::cout<<std::endl;;
        }
        std::cout << ind << operator_ << ":" << std::endl;
        for (const auto& block: node->blocks()) {
            print_block(block, indent + 1);
        }
    }
};

int main(void) {
#if 0
#if 0
    // 1. Inizializziamo il modulo contenitore formale
    torch::jit::Module module("GeneralTrainingFramework");

    // 2. Generazione dei sottografi funzionali ereditati dalle tue utility
    auto user_subgraph = create_user_loss_and_stopping_subgraph();
    auto opt_subgraph  = create_parametrizable_optimizer_subgraph(
        2, OptimizerType::Adam, DecayType::Exponential, 0.95f, 10
    );

    // Iperparametri di configurazione strutturale
    // All'interno del tuo main(void) (Punto 4):
    int max_epochs = 200;
    float learning_rate = 0.01f;
    size_t num_tensors_to_optimize = 2; // Specifichiamo la dimensione reale per l'unrolling interno

    // Chiamata aggiornata e pulita
    auto main_graph = build_universal_training_pipeline(
        user_subgraph, opt_subgraph, num_tensors_to_optimize, max_epochs, learning_rate
    );
    // =========================================================================
    // GESTIONE CRITICA DEL SERIALIZZATORE (Allineamento con le utility stabili)
    // =========================================================================
    // 4. Iniettiamo il nodo obbligatorio "self" all'indice zero della lista degli input
    torch::jit::Value* self_val = main_graph->insertInput(0, "self");
    self_val->setType(module._ivalue()->type()); // Collega il puntatore all'istanza del tipo di classe

    c10::QualifiedName method_name(*module.type()->name(), "optimize_problem");
    auto cu = module._ivalue()->compilation_unit();

    // 5. Costruzione dello Schema Funzionale formale a tipo forte senza tuple anonime radice
    auto tensor_list_type = c10::ListType::ofTensors();

    std::vector<c10::Argument> arguments = {
        c10::Argument("self", module.type()),
        c10::Argument("dataset_inputs", tensor_list_type),
        c10::Argument("dataset_targets", tensor_list_type),
        c10::Argument("initial_weights", tensor_list_type)
    };

    std::vector<c10::Argument> returns = {
        c10::Argument("", tensor_list_type) // Restituisce direttamente List[Tensor]
    };

    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    // 6. Registrazione e compilazione del metodo nella compilation unit
    auto method = cu->create_function(method_name, main_graph);
    method->setSchema(std::move(schema)); // Associa lo schema formale

    // 7. Legatura del metodo al runtime del modulo
    module.type()->addMethod(method);

    // 8. Esportazione finale su disco
    // Disabilita temporaneamente l'ottimizzatore JIT globale
    torch::jit::GraphOptimizerEnabledGuard guard(false);
    // Disattiva il pass di ottimizzazione che fonde o modifica i Loop
    torch::jit::setGraphExecutorOptimize(false);
    std::string filename = "GeneralTrainingFramework.pt";
    module.save(filename);
    std::cout << "SUCCESS: Compilazione della pipeline completata. Modulo salvato in: " << filename << std::endl;
#else
    if (false) {
        // Definiamo un modulo TorchScript inline per testare il flusso di controllo
        auto module_code = R"JIT(
        def test_if_else(input_val: Tensor) -> Tuple[int, Tensor]:
            # 1. Inizializzazione delle variabili da aggiornare
            status = 0
            value = input_val * 2

            for i in range(20):
                # 2. Blocco Condizionale (Entrambi i rami aggiornano ENTRAMBE le variabili)
                value = value * i * 0.05
                if torch.all(value > 0.01):
                    status = 1
                    value = value + 10.0
                else:
                    break

            return status, value
    )JIT";

        // Compilazione del codice TorchScript
        auto cu = torch::jit::compile(module_code);
        auto& function = cu->get_function("test_if_else");
        auto graphR = toGraphFunction(function).graph();

        print_block(graphR->block());

        std::cout << "=== LIBTORCH JIT GRAPH ===" << std::endl;
        toGraphFunction(function).graph()->print(std::cout) << std::endl;
        return 0;


        // Preparazione degli input per il blocco 'then'
        c10::IValue input_tensor = torch::ones({2, 2});

        // Esecuzione caso 'then' (condition = true)
        auto result_then = function({input_tensor}).toTuple();
        std::cout << "--- Blocco THEN (True) ---" << std::endl;
        std::cout << "Status: " << result_then->elements()[0].toInt() << std::endl;
        std::cout << "Value Tensor:\n" << result_then->elements()[1].toTensor() << std::endl;

        // Esecuzione caso 'else' (condition = false)
        auto result_else = function({input_tensor}).toTuple();
        std::cout << "\n--- Blocco ELSE (False) ---" << std::endl;
        std::cout << "Status: " << result_else->elements()[0].toInt() << std::endl;
        std::cout << "Value Tensor:\n" << result_else->elements()[1].toTensor() << std::endl;

        return 0;
    } else if (false) {

 // 1. Inizializzazione corretta della CompilationUnit e della classe del modulo
        // 1. Inizializzazione della CompilationUnit e della classe del modulo
        auto cu = std::make_shared<torch::jit::CompilationUnit>();
    auto class_name = c10::QualifiedName("__torch__.ConditionalModule");
    auto class_type = c10::ClassType::create(class_name, cu, /*is_module=*/true);
    torch::jit::Module module_(cu, class_type);

    auto graph = std::make_shared<torch::jit::Graph>();

    // 1. Input del grafo
    torch::jit::Value* self_val = graph->addInput("self")->setType(class_type);
    torch::jit::Value* condition = graph->addInput("condition")->setType(c10::BoolType::get());
    torch::jit::Value* input_val = graph->addInput("input_val")->setType(c10::TensorType::get());

    // Funzione helper per creare costanti formali e pulite


    torch::jit::Value* const_2 = graph->insertConstant(2)->setType( c10::IntType::get());

    // Costante Alpha formale (deve essere uno scalare intero = 1)
    torch::jit::Value* alpha_1 = graph->insertConstant(1)->setType( c10::IntType::get());

    // 2. Calcolo iniziale: value = input_val * 2
    torch::jit::Node* mul_node = graph->create(torch::jit::aten::mul, {input_val, const_2});
    graph->insertNode(mul_node);
    torch::jit::Value* initial_value = mul_node->output()->setType(c10::TensorType::get());

    // 3. Creazione del nodo condizionale IF
    torch::jit::Node* if_node = graph->create(torch::jit::prim::If, {condition});
    graph->insertNode(if_node);

    torch::jit::Block* then_block = if_node->addBlock();
    torch::jit::Block* else_block = if_node->addBlock();

    // --- BLOCCO THEN (True) ---
    auto then_status_node = graph->create(torch::jit::prim::Constant);
    then_status_node->output()->setType(c10::IntType::get());
    then_status_node->i_(c10::attr::value, 1);
    then_block->appendNode(then_status_node);
    torch::jit::Value* then_status = then_status_node->output();

    auto const_10_node = graph->create(torch::jit::prim::Constant);
    const_10_node->output()->setType(c10::FloatType::get());
    const_10_node->f_(c10::attr::value, 10.0);
    then_block->appendNode(const_10_node);

    // Firma esatta JIT: aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor
    torch::jit::Node* then_add = graph->create(torch::jit::aten::add, {initial_value, const_10_node->output(), alpha_1});
    then_block->appendNode(then_add);
    torch::jit::Value* then_value = then_add->output()->setType(c10::TensorType::get()); // Forza tipo Tensor singolo

    then_block->registerOutput(then_status);
    then_block->registerOutput(then_value);

    // --- BLOCCO ELSE (False) ---
    auto else_status_node = graph->create(torch::jit::prim::Constant);
    else_status_node->output()->setType(c10::IntType::get());
    else_status_node->i_(c10::attr::value, -1);
    else_block->appendNode(else_status_node);
    torch::jit::Value* else_status = else_status_node->output();

    auto const_5_node = graph->create(torch::jit::prim::Constant);
    const_5_node->output()->setType(c10::FloatType::get());
    const_5_node->f_(c10::attr::value, 5.0);
    else_block->appendNode(const_5_node);

    // Firma esatta JIT: aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor
    torch::jit::Node* else_sub = graph->create(torch::jit::aten::sub, {initial_value, const_5_node->output(), alpha_1});
    else_block->appendNode(else_sub);
    torch::jit::Value* else_value = else_sub->output()->setType(c10::TensorType::get()); // Forza tipo Tensor singolo

    else_block->registerOutput(else_status);
    else_block->registerOutput(else_value);

    // 4. Mappatura degli output FISICI sul nodo IF (SOLO 2 OUTPUT)
    // Svuotiamo prima eventuali output spuri ereditati se presenti
    while(if_node->outputs().size() > 0) {
        if_node->eraseOutput(0);
    }
    torch::jit::Value* final_status = if_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* final_value = if_node->addOutput()->setType(c10::TensorType::get());

    // 5. Costruzione della Tupla di ritorno del grafo
    torch::jit::Node* tuple_node = graph->create(torch::jit::prim::TupleConstruct, {final_status, final_value});
    graph->insertNode(tuple_node);
    graph->registerOutput(tuple_node->output());

    // Registrazione formale del tipo della Tupla Nominata
    auto tensor_type = c10::TensorType::get();
    auto int_type = c10::IntType::get();
    auto bool_type = c10::BoolType::get();
    auto tuple_name = c10::QualifiedName("__torch__.MyReturnTypeTuple");
    auto tuple_return_type = c10::TupleType::create({int_type, tensor_type});
    cu->register_type(tuple_return_type);
    tuple_node->output()->setType(tuple_return_type);

    // 6. Schema e Compilazione finale
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", class_type),
        c10::Argument("condition", bool_type),
        c10::Argument("input_val", tensor_type)
    };
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    c10::QualifiedName method_name(class_name, "forward");
    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);
    method->setSchema(std::move(schema));
    class_type->addMethod(method);

    // 10. Salvataggio sicuro (.pt)
    module_.save("conditional_subgraph.pt");
    std::cout << "Modello salvato con successo senza eccezioni JIT!" << std::endl;

    } else {
#if 0
       auto cu = std::make_shared<torch::jit::CompilationUnit>();
    auto class_name = c10::QualifiedName("__torch__.ConditionalLoopModule");
    auto class_type = c10::ClassType::create(class_name, cu, /*is_module=*/true);
    torch::jit::Module module_(cu, class_type);

    auto graph = std::make_shared<torch::jit::Graph>();

    // Definizione formale degli input del grafo (Index 0: self, Index 1: input_val)
    torch::jit::Value* self_val = graph->addInput("self")->setType(class_type);
    torch::jit::Value* input_val = graph->addInput("input_val")->setType(c10::TensorType::get());

    // =========================================================================
    // 2. Costanti del Grafo Principale (Utilizzando la sintassi nativa inline)
    // =========================================================================
    torch::jit::Value* const_max_int = graph->insertConstant(std::numeric_limits<int64_t>::max())->setType(c10::IntType::get()); // %80
    torch::jit::Value* const_true    = graph->insertConstant(true)->setType(c10::BoolType::get());                     // %8
    torch::jit::Value* status_init   = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());                // %status.1
    torch::jit::Value* const_2       = graph->insertConstant((int64_t)2)->setType(c10::IntType::get());                // %3
    torch::jit::Value* const_20      = graph->insertConstant((int64_t)20)->setType(c10::IntType::get());               // %5
    torch::jit::Value* const_1       = graph->insertConstant((int64_t)1)->setType(c10::IntType::get());                // %22
    torch::jit::Value* loop_counter_init = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());            // Contatore iniziale range(20)
    torch::jit::Value* const_false   = graph->insertConstant(false)->setType(c10::BoolType::get());                   // %55

    // Costanti in virgola mobile (FloatType)
    torch::jit::Value* const_0_05    = graph->insertConstant(0.05)->setType(c10::FloatType::get());                    // %13
    torch::jit::Value* const_0_01    = graph->insertConstant(0.01)->setType(c10::FloatType::get());                    // %17
    torch::jit::Value* const_10      = graph->insertConstant(10.0)->setType(c10::FloatType::get());                    // %25

    // 3. Calcolo iniziale esterno al ciclo: value = input_val * 2
    torch::jit::Node* mul_node = graph->create(torch::jit::aten::mul, {input_val, const_2});
    graph->insertNode(mul_node);
    torch::jit::Value* value_init = mul_node->output()->setType(c10::TensorType::get());

    // =========================================================================
    // 4. CREAZIONE DEL NODO PRIM::LOOP
    // Passiamo le 3 dipendenze portate dal ciclo: status_init, value_init, loop_counter_init
    // =========================================================================
    torch::jit::Node* loop_node = graph->create(torch::jit::prim::Loop, {const_max_int, const_true, status_init, value_init, loop_counter_init});
    graph->insertNode(loop_node);

    torch::jit::Block* loop_block = loop_node->addBlock();


    // Configurazione degli input del blocco (Index 0: iterazione JIT implicita, Index 1+: dipendenze)
   loop_block->addInput()->setType(c10::IntType::get()); // Obbligatorio a livello strutturale JIT
    torch::jit::Value* status_curr = loop_block->addInput()->setType(c10::IntType::get());
    torch::jit::Value* value_curr = loop_block->addInput()->setType(c10::TensorType::get());
    torch::jit::Value* counter_curr = loop_block->addInput()->setType(c10::IntType::get());
        std::cout << loop_block->inputs().size() << std::endl;

    // --- LOGICA INTERNA AL LOOP ---
    // value = value * i * 0.05
    torch::jit::Node* mul_i = graph->create(torch::jit::aten::mul, {value_curr, counter_curr});
    loop_block->appendNode(mul_i);
    torch::jit::Node* mul_0_05 = graph->create(torch::jit::aten::mul, {mul_i->output(), const_0_05});
    loop_block->appendNode(mul_0_05);
    torch::jit::Value* value_scaled = mul_0_05->output()->setType(c10::TensorType::get());

    // Condizione: torch::all(value > 0.01)
    torch::jit::Node* gt_node = graph->create(torch::jit::aten::gt, {value_scaled, const_0_01});
    loop_block->appendNode(gt_node);
    torch::jit::Node* all_node = graph->create(torch::jit::aten::all, {gt_node->output()});
    loop_block->appendNode(all_node);
    torch::jit::Node* bool_cast = graph->create(torch::jit::aten::Bool, {all_node->output()});
    loop_block->appendNode(bool_cast);
    torch::jit::Value* if_condition = bool_cast->output()->setType(c10::BoolType::get());

    // =========================================================================
    // 5. CREAZIONE DEL NODO PRIM::IF DENTRO IL LOOP
    // =========================================================================
    torch::jit::Node* if_node = graph->create(torch::jit::prim::If, {if_condition});
    loop_block->appendNode(if_node);

    torch::jit::Block* then_block = if_node->addBlock();
    torch::jit::Block* else_block = if_node->addBlock();

    // Ramo THEN (True): continua=True, status=1, value=value+10.0
    torch::jit::Node* then_add = graph->create(torch::jit::aten::add, {value_scaled, const_10, const_1});
        auto result = then_add->output()->setType(c10::TensorType::get());
    then_block->appendNode(then_add); // required for returning it. THis equates to the declaration of the value to be returned.
    then_block->registerOutput(const_true);
    then_block->registerOutput(const_1);
    then_block->registerOutput(result);

    // Ramo ELSE (False / BREAK): continua=False, status=status_attuale, value=value_attuale
    else_block->registerOutput(const_false);
    else_block->registerOutput(status_curr);
    else_block->registerOutput(value_scaled);

    // Catturiamo i 3 output generati dal blocco condizionale condotto all'esterno
    torch::jit::Value* if_continue = if_node->addOutput()->setType(c10::BoolType::get());
    torch::jit::Value* if_status   = if_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* if_value    = if_node->addOutput()->setType(c10::TensorType::get());

    // Incremento del contatore di ciclo: counter = counter + 1
    torch::jit::Node* next_counter = graph->create(torch::jit::aten::add, {counter_curr, const_1});
    auto counter = next_counter->output()->setType(c10::IntType::get());
    loop_block->appendNode(next_counter);

    // Condizione di arresto basata sulle iterazioni massime: counter < 20
    torch::jit::Node* lt_node = graph->create(torch::jit::aten::lt, {counter, const_20});
       auto ltout = lt_node->output()->setType(c10::BoolType::get());
    loop_block->appendNode(lt_node);

    // Condizione di ciclo complessiva combinata in AND logico: (counter < 20) AND (if_continue)

    torch::jit::Node* and_node = graph->create(torch::jit::aten::__and__, {ltout, if_continue});
        auto andout = and_node->output()->setType(c10::BoolType::get());
    loop_block->appendNode(and_node);
    torch::jit::Value* loop_keep_going = andout;

    // =========================================================================
    // 6. REGISTRAZIONE DEGLI OUTPUT DEL BLOCCO DEL LOOP
    // Index 0: Segnale di continuazione (loop_keep_going)
    // Index 1+: Le 3 dipendenze speculari in uscita (status, value, counter)
    // =========================================================================
    loop_block->registerOutput(loop_keep_going);
    loop_block->registerOutput(if_status);
    loop_block->registerOutput(if_value);
    loop_block->registerOutput(counter);

    // =========================================================================
    // 7. ASSEGNAZIONE DEGLI OUTPUT ESTERNI DEL NODO LOOP
    // Forniamo 3 punti di ancoraggio esterni per raccogliere i risultati finali del ciclo
    // =========================================================================
    torch::jit::Value* final_status  = loop_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* final_value   = loop_node->addOutput()->setType(c10::TensorType::get());
    // torch::jit::Value* final_counter = loop_node->addOutput()->setType(c10::IntType::get()); // Variabile di appoggio interna

        //TEST

       // torch::jit::LoopView stmt(loop_node);
       //  auto loop_type = stmt.loopType();
       //  auto bci = stmt.bodyCarriedInputs();
       //  auto co = stmt.carriedOutputs();

    // =========================================================================
    // 8. COSTRUZIONE DELLA TUPLA DI RITORNO DEL GRAFO PRINCIPALE
    // =========================================================================
    torch::jit::Node* tuple_node = graph->create(torch::jit::prim::TupleConstruct, {final_status, final_value});
    graph->insertNode(tuple_node);
    graph->registerOutput(tuple_node->output());

    // Registrazione formale del tipo della Tupla Nominata all'interno del gestore tipi della CompilationUnit
    auto tensor_type = c10::TensorType::get();
    auto int_type = c10::IntType::get();
    auto bool_type = c10::BoolType::get();
    auto tuple_name = c10::QualifiedName("__torch__.MyReturnTypeTuple");
    auto tuple_return_type = c10::TupleType::create({int_type, tensor_type});
    cu->register_type(tuple_return_type);
    tuple_node->output()->setType(tuple_return_type);

    // =========================================================================
    // 9. DEFINIZIONE DELLO SCHEMA REALE E COMPILAZIONE FINALE DEL METODO
    // =========================================================================
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", class_type),
        c10::Argument("input_val", tensor_type)
    };
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    c10::QualifiedName method_name(class_name, "forward");
    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    // Generazione del metodo di runtime nell'unità di compilazione
    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);
    method->setSchema(std::move(schema));
    class_type->addMethod(method);
        module_.save("conditional_subgraph.pt");
        graph->print(std::cout);
#else

#endif
    }
#endif

#else
auto cu = std::make_shared<torch::jit::CompilationUnit>();
    auto class_name = c10::QualifiedName("__torch__.ConditionalLoopModule");
    auto class_type = c10::ClassType::create(class_name, cu, /*is_module=*/true);
    torch::jit::Module module_(cu, class_type);

    auto graph = std::make_shared<torch::jit::Graph>();

    // 1. Input del grafo
    torch::jit::Value* self_val = graph->addInput("self")->setType(class_type);
    torch::jit::Value* input_val = graph->addInput("input_val")->setType(c10::TensorType::get());

    // =========================================================================
    // 2. Costanti del Grafo Principale
    // =========================================================================
    torch::jit::Value* const_max_int = graph->insertConstant(std::numeric_limits<int64_t>::max())->setType(c10::IntType::get()); // %80
    torch::jit::Value* const_true    = graph->insertConstant(true)->setType(c10::BoolType::get());                     // %8
    torch::jit::Value* status_init   = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());                // %status.1
    torch::jit::Value* const_2       = graph->insertConstant((int64_t)2)->setType(c10::IntType::get());                // %3
    torch::jit::Value* const_20      = graph->insertConstant((int64_t)20)->setType(c10::IntType::get());               // %5
    torch::jit::Value* const_1       = graph->insertConstant((int64_t)1)->setType(c10::IntType::get());                // %22
    torch::jit::Value* loop_counter_init = graph->insertConstant((int64_t)0)->setType(c10::IntType::get());            // Contatore iniziale range(20)
    torch::jit::Value* const_false   = graph->insertConstant(false)->setType(c10::BoolType::get());                   // %55

    // Costanti in virgola mobile
    torch::jit::Value* const_0_05    = graph->insertConstant(0.05)->setType(c10::FloatType::get());                    // %13
    torch::jit::Value* const_0_01    = graph->insertConstant(0.01)->setType(c10::FloatType::get());                    // %17
    torch::jit::Value* const_10      = graph->insertConstant(10.0)->setType(c10::FloatType::get());                    // %25

    // 3. Calcolo iniziale esterno al ciclo: value = input_val * 2
    torch::jit::Node* mul_node = graph->create(torch::jit::aten::mul, {input_val, const_2});
    graph->insertNode(mul_node);
    torch::jit::Value* value_init = mul_node->output()->setType(c10::TensorType::get());

    // =========================================================================
    // 4. CREAZIONE DEL NODO PRIM::LOOP
    // Per seguire lo standard nativo esatto del compilatore PyTorch JIT:
    // Ordine input: max_trip, cond_in, status_init, value_init, counter_init (status_init viene riutilizzato come contatore nel grafo target)
    // =========================================================================
    torch::jit::Node* loop_node = graph->create(torch::jit::prim::Loop, {const_max_int, const_true, status_init, value_init, status_init});
    graph->insertNode(loop_node);

    torch::jit::Block* loop_block = loop_node->addBlock();

    // Input del blocco nell'ordine IR nativo
    loop_block->addInput()->setType(c10::IntType::get()); // Iterazione JIT implicita
    torch::jit::Value* status_curr = loop_block->addInput()->setType(c10::IntType::get());   // Int
    torch::jit::Value* value_curr  = loop_block->addInput()->setType(c10::TensorType::get());  // Tensor
    torch::jit::Value* counter_curr = loop_block->addInput()->setType(c10::IntType::get()); // Int (il contatore %82 del grafo target)

    // --- LOGICA INTERNA AL LOOP ---
    // value = value * (i+1) * 0.05
    // Incremento del contatore di ciclo
    torch::jit::Node* next_counter = graph->create(torch::jit::aten::add, {counter_curr, const_1});
    loop_block->appendNode(next_counter);
    torch::jit::Value* counter = next_counter->output()->setType(c10::IntType::get());
    torch::jit::Node* mul_i = graph->create(torch::jit::aten::mul, {value_curr, counter});
    loop_block->appendNode(mul_i);
    torch::jit::Node* mul_0_05 = graph->create(torch::jit::aten::mul, {mul_i->output(), const_0_05});
    loop_block->appendNode(mul_0_05);
    torch::jit::Value* value_scaled = mul_0_05->output()->setType(c10::TensorType::get());

    // Condizione: torch::all(value > 0.01)
    torch::jit::Node* gt_node = graph->create(torch::jit::aten::gt, {value_scaled, const_0_01});
    loop_block->appendNode(gt_node);
    torch::jit::Node* all_node = graph->create(torch::jit::aten::all, {gt_node->output()});
    loop_block->appendNode(all_node);
    torch::jit::Node* bool_cast = graph->create(torch::jit::aten::Bool, {all_node->output()});
    loop_block->appendNode(bool_cast);
    torch::jit::Value* if_condition = bool_cast->output()->setType(c10::BoolType::get());

    // =========================================================================
    // 5. CREAZIONE DEL NODO PRIM::IF DENTRO IL LOOP
    // =========================================================================
    torch::jit::Node* if_node = graph->create(torch::jit::prim::If, {if_condition});
    loop_block->appendNode(if_node);

    torch::jit::Block* then_block = if_node->addBlock();
    torch::jit::Block* else_block = if_node->addBlock();

    // Ramo THEN (True) -> output: (continua, status, value)
    torch::jit::Node* then_add = graph->create(torch::jit::aten::add, {value_scaled, const_10, const_1});
    then_block->appendNode(then_add);
    torch::jit::Value* result = then_add->output()->setType(c10::TensorType::get());

    then_block->registerOutput(const_true);
    then_block->registerOutput(const_1);
    then_block->registerOutput(result);

    // Ramo ELSE (False / BREAK) -> output: (continua, status, value)
    else_block->registerOutput(const_false);
    else_block->registerOutput(status_curr);
    else_block->registerOutput(value_scaled);

    // -------------------------------------------------------------------------
    // CRITICAL FIX: Svuotiamo e forziamo ESATTAMENTE 3 output sul nodo prim::If
    // Questo elimina il quarto registro fantasma (_10) nel codice Python!
    // -------------------------------------------------------------------------
    while (if_node->outputs().size() > 0) {
        if_node->eraseOutput(0);
    }

    torch::jit::Value* if_continue = if_node->addOutput()->setType(c10::BoolType::get());
    torch::jit::Value* if_status   = if_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* if_value    = if_node->addOutput()->setType(c10::TensorType::get());
    // -------------------------------------------------------------------------



    // Condizione di arresto basata sulle iterazioni massime
    torch::jit::Node* lt_node = graph->create(torch::jit::aten::lt, {counter, const_20});
    loop_block->appendNode(lt_node);
    torch::jit::Value* ltout = lt_node->output()->setType(c10::BoolType::get());

    // Condizione logica complessiva: (counter < 20) AND (if_continue)
    torch::jit::Node* and_node = graph->create(torch::jit::aten::__and__, {ltout, if_continue});
    loop_block->appendNode(and_node);
    torch::jit::Value* loop_keep_going = and_node->output()->setType(c10::BoolType::get());


    // =========================================================================
    // 6. REGISTRAZIONE DEGLI OUTPUT DEL BLOCCO DEL LOOP
    // =========================================================================
    loop_block->registerOutput(loop_keep_going);
    loop_block->registerOutput(if_status);
    loop_block->registerOutput(if_value);
    loop_block->registerOutput(counter);

    // =========================================================================
    // 7. ASSEGNAZIONE DEGLI OUTPUT ESTERNI DEL NODO LOOP
    // Svuotiamo preventivamente e forziamo la tipizzazione esatta per allineare l'assegnazione Python
    // =========================================================================
    while (loop_node->outputs().size() > 0) {
        loop_node->eraseOutput(0);
    }
    torch::jit::Value* final_status  = loop_node->addOutput()->setType(c10::IntType::get());
    torch::jit::Value* final_value   = loop_node->addOutput()->setType(c10::TensorType::get());
    torch::jit::Value* final_counter = loop_node->addOutput()->setType(c10::IntType::get());

    // =========================================================================
    // 8. COSTRUZIONE DELLA TUPLA DI RITORNO DEL GRAFO PRINCIPALE
    // =========================================================================
    torch::jit::Node* tuple_node = graph->create(torch::jit::prim::TupleConstruct, {final_status, final_value});
    graph->insertNode(tuple_node);
    graph->registerOutput(tuple_node->output());

    auto tensor_type = c10::TensorType::get();
    auto int_type = c10::IntType::get();
    auto tuple_name = c10::QualifiedName("__torch__.MyReturnTypeTuple");
    auto tuple_return_type = c10::TupleType::create({int_type, tensor_type});
    cu->register_type(tuple_return_type);
    tuple_node->output()->setType(tuple_return_type);

    // =========================================================================
    // 9. SCHEMA E COMPILAZIONE
    // =========================================================================
    std::vector<c10::Argument> arguments = {
        c10::Argument("self", class_type),
        c10::Argument("input_val", tensor_type)
    };
    std::vector<c10::Argument> returns = {
        c10::Argument("", tuple_return_type)
    };

    c10::QualifiedName method_name(class_name, "forward");
    c10::FunctionSchema schema(method_name.name(), "", arguments, returns);

    auto method = module_._ivalue()->compilation_unit()->create_function(method_name, graph);
    method->setSchema(std::move(schema));
    class_type->addMethod(method);

    module_.save("conditional_subgraph.pt");
    std::cout << "Modello salvato." << std::endl;

    return 0;
#endif

}
