{-|
Module      : Idris.REPL
Description : Main function to decide Idris' mode of use.
License     : BSD3
Maintainer  : The Idris Community.
-}
module Main
  (main)
  where

import Idris.AbsSyntax
import Idris.Core.CaseTree
import Idris.Core.Evaluate
import Idris.Core.TT
import Idris.Delaborate
import Idris.Docs
import Idris.ElabDecls
import Idris.Main (runMain)
import Idris.ModeCommon
import Idris.Output
import Idris.REPL.Browse
--
import Control.Category
import Control.DeepSeq
import Control.Monad
import Data.Maybe
import Language.C hiding (Name)
-- import Language.C.Data.Ident
import Language.C.Syntax.AST
import Language.C.System.GCC
import Prelude hiding (id, (.), (<$>))
import System.Environment (getArgs)
import System.IO
import Safe
import Text.Groom
import Text.PrettyPrint.Annotated.Leijen hiding ((</>))
import Text.PrettyPrint.HughesPJ (render)

-- | The main function for the executable.
-- Main program reads command line options, parses the main program, and gets
-- on with the REPL.
-- | The main function of Idris that when given a set of Options will
-- launch Idris into the desired interaction mode either: REPL;
-- Compiler; Script execution; or IDE Mode.
-- idrisMain :: [Opt] -> Idris ()
-- idrisMain opts =
--   do   mapM_ setWidth (opt getConsoleWidth opts)
--        let inputs = opt getFile opts
--        let quiet = Quiet `elem` opts
--        let nobanner = NoBanner `elem` opts
--        let idesl = Idemode `elem` opts || IdemodeSocket `elem` opts
--        let runrepl = not (NoREPL `elem` opts)
--        let output = opt getOutput opts
--        let ibcsubdir = opt getIBCSubDir opts
--        let importdirs = opt getImportDir opts
--        let sourcedirs = opt getSourceDir opts
--        setSourceDirs sourcedirs
--        let bcs = opt getBC opts
--        let pkgdirs = opt getPkgDir opts
--        -- Set default optimisations
--        let optimise = case opt getOptLevel opts of
--                         [] -> 2
--                         xs -> last xs
--        setOptLevel optimise
--        let outty = case opt getOutputTy opts of
--                      [] -> if Interface `elem` opts then
--                               Object else Executable
--                      xs -> last xs
--        let cgn = case opt getCodegen opts of
--                    [] -> Via IBCFormat "c"
--                    xs -> last xs
--        let cgFlags = opt getCodegenArgs opts
--        -- Now set/unset specifically chosen optimisations
--        let os = opt getOptimisation opts
--        mapM_ processOptimisation os
--        script <- case opt getExecScript opts of
--                    []     -> return Nothing
--                    x:y:xs -> do iputStrLn "More than one interpreter expression found."
--                                 runIO $ exitWith (ExitFailure 1)
--                    [expr] -> return (Just expr)
--        let immediate = opt getEvalExpr opts
--        let port = case getPort opts of
--                     Nothing -> ListenPort defaultPort
--                     Just p  -> p
--        when (DefaultTotal `elem` opts) $ do i <- getIState
--                                             putIState (i { default_total = DefaultCheckingTotal })
--        tty <- runIO isATTY
--        setColourise $ not quiet && last (tty : opt getColour opts)
--        mapM_ addLangExt (opt getLanguageExt opts)
--        setREPL runrepl
--        setQuiet (quiet || isJust script || not (null immediate))
--        setCmdLine opts
--        setOutputTy outty
--        setNoBanner nobanner
--        setCodegen cgn
--        mapM_ (addFlag cgn) cgFlags
--        mapM_ makeOption opts
--        vlevel <- verbose
--        when (runrepl && vlevel == 0) $ setVerbose 1
--        -- if we have the --bytecode flag, drop into the bytecode assembler
--        case bcs of
--          [] -> return ()
--          xs -> return () -- runIO $ mapM_ bcAsm xs
--        case ibcsubdir of
--          [] -> setIBCSubDir ""
--          (d:_) -> setIBCSubDir d
--        setImportDirs importdirs
--        setNoBanner nobanner
--        when (not (NoBasePkgs `elem` opts)) $ do
--            addPkgDir "prelude"
--            addPkgDir "base"
--        mapM_ addPkgDir pkgdirs
--        elabPrims
--        when (not (NoBuiltins `elem` opts)) $ do x <- loadModule "Builtins" (IBC_REPL True)
--                                                 addAutoImport "Builtins"
--                                                 return ()
--        when (not (NoPrelude `elem` opts)) $ do x <- loadModule "Prelude" (IBC_REPL True)
--                                                addAutoImport "Prelude"
--                                                return ()
--        when (runrepl && not idesl) initScript
--        nobanner <- getNoBanner
--        when (runrepl &&
--              not quiet &&
--              not idesl &&
--              not (isJust script) &&
--              not nobanner &&
--              null immediate) $
--          iputStrLn banner
--        orig <- getIState
--        mods <- if idesl then return [] else loadInputs inputs Nothing
--        let efile = case inputs of
--                         [] -> ""
--                         (f:_) -> f
--        runIO $ hSetBuffering stdout LineBuffering
--        ok <- noErrors
--        when ok $ case output of
--                     [] -> return ()
--                     (o:_) -> idrisCatch (process "" (Compile cgn o))
--                                (\e -> do ist <- getIState ; iputStrLn $ pshow ist e)
--        case immediate of
--          [] -> return ()
--          exprs -> do setWidth InfinitelyWide
--                      mapM_ (\str -> do ist <- getIState
--                                        c <- colourise
--                                        case parseExpr ist str of
--                                          Failure (ErrInfo err _) -> do iputStrLn $ show (fixColour c err)
--                                                                        runIO $ exitWith (ExitFailure 1)
--                                          Success e -> process "" (Eval e))
--                            exprs
--                      runIO exitSuccess
--        case script of
--          Nothing -> return ()
--          Just expr -> execScript expr
--        -- Create Idris data dir + repl history and config dir
--        idrisCatch (do dir <- runIO $ getIdrisUserDataDir
--                       exists <- runIO $ doesDirectoryExist dir
--                       unless exists $ logLvl 1 ("Creating " ++ dir)
--                       runIO $ createDirectoryIfMissing True (dir </> "repl"))
--          (\e -> return ())
--        historyFile <- runIO $ getIdrisHistoryFile
--        when ok $ case opt getPkgIndex opts of
--                       (f : _) -> writePkgIndex f
--                       _ -> return ()
--        when (runrepl && not idesl) $ do
-- --          clearOrigPats
--          case port of
--            DontListen -> return ()
--            ListenPort port' -> startServer port' orig mods
--          runInputT (replSettings (Just historyFile)) $ repl (force orig) mods efile
--        let idesock = IdemodeSocket `elem` opts
--        when (idesl) $ idemodeStart idesock orig inputs
--        ok <- noErrors
--        when (not ok) $ runIO (exitWith (ExitFailure 1))
--   where
--     makeOption (OLogging i)     = setLogLevel i
--     makeOption (OLogCats cs)    = setLogCats cs
--     makeOption (Verbose v)      = setVerbose v
--     makeOption TypeCase         = setTypeCase True
--     makeOption TypeInType       = setTypeInType True
--     makeOption NoCoverage       = setCoverage False
--     makeOption ErrContext       = setErrContext True
--     makeOption (IndentWith n)   = setIndentWith n
--     makeOption (IndentClause n) = setIndentClause n
--     makeOption _                = return ()
--     processOptimisation :: (Bool,Optimisation) -> Idris ()
--     processOptimisation (True,  p) = addOptimise p
--     processOptimisation (False, p) = removeOptimise p
--     addPkgDir :: String -> Idris ()
--     addPkgDir p = do ddir <- runIO getIdrisLibDir
--                      addImportDir (ddir </> p)
--                      addIBC (IBCImportDir (ddir </> p))
-- -- | Invoke as if from command line. It is an error if there are
-- -- unresolved totality problems.
-- idris :: [Opt] -> IO (Maybe IState)
-- idris opts = do res <- runExceptT $ execStateT totalMain idrisInit
--                 case res of
--                   Left err -> do putStrLn $ pshow idrisInit err
--                                  return Nothing
--                   Right ist -> return (Just ist)
--     where totalMain = do idrisMain opts
--                          ist <- getIState
--                          case idris_totcheckfail ist of
--                            ((fc, msg):_) -> ierror . At fc . Msg $ "Could not build: "++  msg
--                            [] -> return ()
-- -- | Execute the provided Idris expression.
-- execScript :: String -> Idris ()
-- execScript expr = do i <- getIState
--                      c <- colourise
--                      case parseExpr i expr of
--                           Failure (ErrInfo err _) -> do iputStrLn $ show (fixColour c err)
--                                                         runIO $ exitWith (ExitFailure 1)
--                           Success term -> do ctxt <- getContext
--                                              (tm, _) <- elabVal (recinfo (fileFC "toplevel")) ERHS term
--                                              res <- execute tm
--                                              runIO $ exitSuccess
-- -- | Run the initialisation script
-- initScript :: Idris ()
-- initScript = do script <- runIO $ getIdrisInitScript
--                 idrisCatch (do go <- runIO $ doesFileExist script
--                                when go $ do
--                                  h <- runIO $ openFile script ReadMode
--                                  runInit h
--                                  runIO $ hClose h)
--                            (\e -> iPrintError $ "Error reading init file: " ++ show e)
--     where runInit :: Handle -> Idris ()
--           runInit h = do eof <- lift . lift $ hIsEOF h
--                          ist <- getIState
--                          unless eof $ do
--                            line <- runIO $ hGetLine h
--                            script <- runIO $ getIdrisInitScript
--                            c <- colourise
--                            processLine ist line script c
--                            runInit h
--           processLine i cmd input clr =
--               case parseCmd i input cmd of
--                    Failure (ErrInfo err _) -> runIO $ print (fixColour clr err)
--                    Success (Right Reload) -> iPrintError "Init scripts cannot reload the file"
--                    Success (Right (Load f _)) -> iPrintError "Init scripts cannot load files"
--                    Success (Right (ModImport f)) -> iPrintError "Init scripts cannot import modules"
--                    Success (Right Edit) -> iPrintError "Init scripts cannot invoke the editor"
--                    Success (Right Proofs) -> proofs i
--                    Success (Right Quit) -> iPrintError "Init scripts cannot quit Idris"
--                    Success (Right cmd ) -> process [] cmd
--                    Success (Left err) -> runIO $ print err

main
    :: IO ()
main = do
    args <- getArgs
    putStrLn ("command line arguments: " ++ show args)
    mapM_ (runMain . translateFile) args -- Launch REPL or compile mode.
--     ast <- loadAst
    -- pretty print
--     printMyAST
--         ast
--     printMyAST testAst

-- testAst = CTranslUnit [CDeclExt (CDecl [ ] [(Nothing, Nothing, Nothing)] undefNode) ] undefNode
testAst
    :: CTranslationUnit NodeInfo
testAst =
    CTranslUnit
        [(enum "Distancetype" ["Mile", "Kilometre", "Lightyear"])]
        undefNode

enum :: String -> [String] -> CExternalDeclaration NodeInfo
enum name elements =
    CDeclExt
        (CDecl
             [ CStorageSpec (CTypedef undefNode)
             , CTypeSpec
                   (CEnumType
                        (CEnum
                             Nothing
                             (Just
                                  (map
                                       (\e ->
                                             (internalIdent e, Nothing))
                                       elements))
                             []
                             undefNode)
                        undefNode)]
             [ ( Just
                     (CDeclr
                          (Just (internalIdent name))
                          []
                          Nothing
                          []
                          undefNode)
               , Nothing
               , Nothing)]
             undefNode)

emptyTypedefStruct :: String -> CExternalDeclaration NodeInfo
emptyTypedefStruct name =
    CDeclExt
       (CDecl
          [CStorageSpec
             (CTypedef
                undefNode)
          ,CTypeSpec
             (CSUType
                (CStruct
                   CStructTag
                   (Just
                      (internalIdent
                         name))
                   Nothing
                   []
                   undefNode)
                undefNode)]
          [(Just
              (CDeclr
                 (Just
                    (internalIdent
                       name))
                 []
                 Nothing
                 []
                 undefNode)
           ,Nothing
           ,Nothing)]
          undefNode)

dataConstructors =
  CTranslUnit
    [(enum "Distancetype" ["Angstrom", "Mile", "Kilometre", "Lightyear"])
    -- typedef struct Distance Distance;
    ,emptyTypedefStruct "Distance"
    ,CDeclExt
       (CDecl
          [CStorageSpec
             (CTypedef
                undefNode)
          ,CTypeSpec
             (CSUType
                (CStruct
                   CStructTag
                   (Just
                      (internalIdent
                         "Distance"))
                   (Just
                      [CDecl
                         [CTypeSpec
                            (CTypeDef
                               (internalIdent
                                  "Distancetype")
                               undefNode)]
                         [(Just
                             (CDeclr
                                (Just
                                   (internalIdent
                                      "tag"))
                                []
                                Nothing
                                []
                                undefNode)
                          ,Nothing
                          ,Nothing)]
                         undefNode
                      ,CDecl
                         [CTypeSpec
                            (CSUType
                               (CStruct
                                  CUnionTag
                                  Nothing
                                  (Just
                                     [CDecl
                                        [CTypeSpec
                                           (CDoubleType
                                              undefNode)]
                                        [(Just
                                            (CDeclr
                                               (Just
                                                  (internalIdent
                                                     "angstorms"))
                                               []
                                               Nothing
                                               []
                                               undefNode)
                                         ,Nothing
                                         ,Nothing)]
                                        undefNode
                                     ,CDecl
                                        [CTypeSpec
                                           (CDoubleType
                                              undefNode)]
                                        [(Just
                                            (CDeclr
                                               (Just
                                                  (internalIdent
                                                     "miles"))
                                               []
                                               Nothing
                                               []
                                               undefNode)
                                         ,Nothing
                                         ,Nothing)]
                                        undefNode
                                     ,CDecl
                                        [CTypeSpec
                                           (CDoubleType
                                              undefNode)]
                                        [(Just
                                            (CDeclr
                                               (Just
                                                  (internalIdent
                                                     "kilometres"))
                                               []
                                               Nothing
                                               []
                                               undefNode)
                                         ,Nothing
                                         ,Nothing)]
                                        undefNode
                                     ,CDecl
                                        [CTypeSpec
                                           (CSUType
                                              (CStruct
                                                 CStructTag
                                                 Nothing
                                                 (Just
                                                    [CDecl
                                                       [CTypeSpec
                                                          (CDoubleType
                                                             undefNode)]
                                                       [(Just
                                                           (CDeclr
                                                              (Just
                                                                 (internalIdent
                                                                    "x"))
                                                              []
                                                              Nothing
                                                              []
                                                              undefNode)
                                                        ,Nothing
                                                        ,Nothing)]
                                                       undefNode
                                                    ,CDecl
                                                       [CTypeSpec
                                                          (CDoubleType
                                                             undefNode)]
                                                       [(Just
                                                           (CDeclr
                                                              (Just
                                                                 (internalIdent
                                                                    "y"))
                                                              []
                                                              Nothing
                                                              []
                                                              undefNode)
                                                        ,Nothing
                                                        ,Nothing)]
                                                       undefNode])
                                                 []
                                                 undefNode)
                                              undefNode)]
                                        [(Just
                                            (CDeclr
                                               (Just
                                                  (internalIdent
                                                     "lightyear"))
                                               []
                                               Nothing
                                               []
                                               undefNode)
                                         ,Nothing
                                         ,Nothing)]
                                        undefNode])
                                  []
                                  undefNode)
                               undefNode)]
                         []
                         undefNode])
                   []
                   undefNode)
                undefNode)]
          []
          undefNode)]
    undefNode

loadAst =
    errorOnLeftM "Parse Error" $
    parseCFile (newGCC "gcc") Nothing [] "example.c"

errorOnLeft
    :: (Show a)
    => String -> (Either a b) -> IO b
errorOnLeft msg = either (error . ((msg ++ ": ") ++) . show) return

errorOnLeftM
    :: (Show a)
    => String -> IO (Either a b) -> IO b
errorOnLeftM msg action = action >>= errorOnLeft msg

printMyAST :: CTranslUnit -> IO ()
printMyAST ctu = (print . pretty) ctu

data Callee = Callee
  { cName :: Name
    -- type of arguments that the callee is being
    -- called with.
    -- For example, when Main.MyCons Integer 10 (Main.MyList Integer)
    -- store Integer, Integer and (Main.MyList Integer) here
  , cReturnType :: Type
  -- Idris/Core/TT.hs:type Term = TT Name
  , cArguments :: [Term]
  } deriving (Eq, Show)

translateFile :: String -> Idris ()
translateFile filename = do
  elabPrims
  --      orig <- getIState
  --      clearErr
  let initialState = idrisInit
  putIState $!! initialState
  mods <- loadInputs [filename] Nothing
  (runIO . putStrLn . show) mods
  -- Report either success or failure
  ist <- getIState
  case (errSpan ist) of
        Nothing -> runIO . putStrLn $ "no errors"
        Just x ->
            iPrintError $ "didn't load " ++ filename ++ ", error: " ++ show x
  underNSs <- namespacesInNS []
  (runIO . putStrLn . show) underNSs
    --      mapM_ translateNameSpace underNSs
  names <- namesInNS ["Prelude","Applicative"]
  translateMain
  (mapM_ translateNamedObject . filter ((==) "pure" . show)) names
  let n = lookupTyNameExact (sNS (sUN "main") ["Main"]) (tt_ctxt ist)
  (runIO . putStrLn . show) n
  let s = lookupTyNameExact (sNS (sUN "pure") ["Applicative","Prelude"]) (tt_ctxt ist)
  (runIO . putStrLn . show) s
  callees <- mainCalls
    --      names <- namesInNS []
    --      (runIO . putStrLn . show) ist
    --      names <- namesInNS ["Prelude","Applicative"]
    --      names <- namesInNS ns
    --      if null names
    --         then iPrintError "Invalid or empty namespace"
    --         else do ist <- getIState
    --                 iRenderResult $
    --                   text "Names:" <$>
    --                   indent 2 (vsep (map (\n -> prettyName True False [] n <+> colon <+>
    --                                              (group . align $ pprintDelabTy ist n))
    --                                       names))
  (runIO . putStrLn . groom) callees
  cdecls <- (fmap concat . mapM translateCalleeToC) callees
  (outputAST . toCProgram) cdecls
  return ()

toCProgram :: [CExternalDeclaration NodeInfo] -> CTranslationUnit NodeInfo
toCProgram decls = CTranslUnit decls undefNode

translateCalleeToC :: Callee -> Idris ([CExternalDeclaration NodeInfo])
translateCalleeToC c = getIState >>= translateCalleeToCS c

translateCalleeToCS :: Callee -> IState -> Idris [CExternalDeclaration NodeInfo]
translateCalleeToCS c@(Callee name _ _) istate
  | isRecord name istate =
    translateCalleeRecordToC c istate (lookupCtxtExact name (idris_records istate))
  | isData name istate =
    translateCalleeDataToC c istate (lookupCtxtExact name (idris_datatypes istate))
  | otherwise =
    translateCalleeFunctionToC c istate
      ((lookupCtxtExact name . definitions . tt_ctxt) istate)

isRecord :: Name -> IState -> Bool
isRecord name = isJust . lookupCtxtExact name . idris_records

isData :: Name -> IState -> Bool
isData name = isJust . lookupCtxtExact name . idris_datatypes

translateCalleeRecordToC :: Callee -> IState -> Maybe RecordInfo -> Idris [CExternalDeclaration NodeInfo]
translateCalleeRecordToC _ _ Nothing = return []
translateCalleeRecordToC (Callee name returnType arguments) istate (Just record) = return []

translateCalleeDataToC :: Callee -> IState -> Maybe TypeInfo -> Idris [CExternalDeclaration NodeInfo]
translateCalleeDataToC _ _ Nothing = return []
translateCalleeDataToC (Callee name returnType arguments) istate (Just dataInfo) = return []

translateCalleeFunctionToC :: Callee -> IState -> Maybe TTDecl -> Idris [CExternalDeclaration NodeInfo]
translateCalleeFunctionToC _ _ Nothing = return []
translateCalleeFunctionToC (Callee name returnType arguments) istate (Just (def,_,_,_,_,_)) = do
  docs <- getDocs name FullDocs
  return []

mainCalls :: Idris [Callee]
mainCalls = do
  ist <- getIState
  calls <- (calleesOf . lookupTyNameExact (sNS (sUN "main") ["Main"]) . tt_ctxt) ist
  return (removeDuplicates calls)

calleesOf :: Maybe (Name, Type) -> Idris [Callee]
calleesOf Nothing = return []
calleesOf (Just (name,returnType)) = do
  ist <- getIState
  (runIO . putStrLn . show) name
--   callees <- (fmap concat . mapM calleesFromTTDecl)
--               (lookupCtxt name ((definitions . tt_ctxt) ist))
  callees <- calleesFromTTDecl
              (lookupCtxtExact name ((definitions . tt_ctxt) ist))
  return ([Callee name returnType []] ++ callees)

-- defined in Core/Evaluate.hs
-- type TTDecl = (Def, RigCount, Injectivity, Accessibility, Totality, MetaInformation)
calleesFromTTDecl :: Maybe TTDecl -> Idris [Callee]
calleesFromTTDecl Nothing = return []
calleesFromTTDecl (Just (def,_,_,_,_,_)) = calleesFromDef def

calleesFromDef :: Def -> Idris [Callee]
calleesFromDef (Function ty tm) = error "calleeFromDef Function not defined"
calleesFromDef (TyDecl nt ty) = error "calleeFromDef TyDecl not defined"
calleesFromDef (Operator ty _ _) = error "calleeFromDef Operator not defined"
calleesFromDef (CaseOp _ returnType _ _ _ caseDefs) = do
  returnTypeCallees <- calleesFromTerm returnType
  let (_,sc) = cases_runtime caseDefs
  scCallees <- calleesFromSC sc
  return (returnTypeCallees ++ scCallees)

toCallee :: Name -> Idris (Maybe Callee)
toCallee name =
  getIState >>=
  (return . fmap (\(n,t) -> Callee n t []) . lookupTyNameExact name . tt_ctxt)

-- Idris/Core/TT.hs:type Term = TT Name
calleeFromTerm :: Term -> Idris (Maybe Callee)
-- can only get the name from the P constructor
-- as the type is erased in the P data constructor
calleeFromTerm (P _ n _) = toCallee n
calleeFromTerm (V i) = error "CalleeFromTerm V not defined"
calleeFromTerm (Bind n b t) = error "CalleeFromTerm Bind not defined"
calleeFromTerm (App _ f a) = do
  maybeCallee <- calleeFromTerm f
  return (fmap (\c -> c{cArguments = cArguments c ++ [a]}) maybeCallee)
-- calleesFromTerm (Proj t i) = error "CalleesFromTerm Proj not defined"
-- calleesFromTerm (Constant c) = calleesFromConstant c
-- calleesFromTerm Erased = return []
-- calleesFromTerm Impossible = return []
-- calleesFromTerm (Inferred t) = do
--     (runIO . putStrLn) "Inferred "
--     translateTerm t
-- calleesFromTerm (TType i) = do
--     (runIO . putStrLn) $ "TType " ++ show i
-- calleesFromTerm (UType u) = do
--     (runIO . putStrLn) $ "UType " ++ show u
calleeFromTerm _ = error "CalleeFromTerm not defined"

-- Idris/Core/TT.hs:type Term = TT Name
calleesFromTerm :: Term -> Idris [Callee]
-- can only get the name from the P constructor
-- as the type is erased in the P data constructor
calleesFromTerm p@(P _ n _) = do
  calleeFromTerm p >>= return . maybe [] (\c -> [c])
calleesFromTerm (V i) = error "CalleesFromTerm V not defined"
calleesFromTerm (Bind n b t) = error "CalleesFromTerm Bind not defined"
calleesFromTerm (App _ f a) = do
  maybeCallee <- calleeFromTerm f
  let uMaybeCallee = fmap (\c -> c{cArguments = cArguments c ++ [a]}) maybeCallee
  argCallees <- calleesFromArguments uMaybeCallee
  return (maybe [] (\c -> c : argCallees) uMaybeCallee)
calleesFromTerm c@(Constant _) = return ([Callee (sUN "Constant") c []])
calleesFromTerm t = error ( "CalleesFromTerm not defined: " ++ show t)
-- calleesFromTerm (Proj t i) = error "CalleesFromTerm Proj not defined"
-- calleesFromTerm Erased = return []
-- calleesFromTerm Impossible = return []
-- calleesFromTerm (Inferred t) = do
--     (runIO . putStrLn) "Inferred "
--     translateTerm t
-- calleesFromTerm (TType i) = do
--     (runIO . putStrLn) $ "TType " ++ show i
-- calleesFromTerm (UType u) = do
--     (runIO . putStrLn) $ "UType " ++ show u

calleesFromArguments :: Maybe Callee -> Idris [Callee]
calleesFromArguments Nothing = return []
calleesFromArguments (Just callee) = (fmap concat . mapM calleesFromTerm . cArguments) callee

calleesFromSC :: SC -> Idris [Callee]
calleesFromSC (STerm tm) = calleesFromTerm tm
calleesFromSC sc = error ( "calleesFromSC pattern not captured: " ++ show sc)

translateMain :: Idris ()
translateMain = do
  namesInMain <- namesInNS ["Main"]
  --   mapM_ translateNamedObject namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.main" . show))
       namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.MkPerson" . show))
       namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.MyNil" . show))
       namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.Distance" . show))
        namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.Mile" . show))
        namesInMain
  (mapM_ translateNamedObject . filter ((==) "Main.Person" . show))
        namesInMain
  (mapM_ translateDataType . filter ((==) "Main.Distance" . show))
        namesInMain
  (mapM_ translateDataType . filter ((==) "Main.Pair" . show))
        namesInMain
  (mapM_ translateNamedObject . filter ((==) "String" . show))
        namesInMain

translateDataType n = do
  i <- getIState
--   jri <- lookupCtxtExact n (idris_records i)
--   maybe "" ( ) jri
  let jti = lookupCtxtExact n (idris_datatypes i)
  (runIO . putStrLn) $ maybe "" (show . con_names) jti

translateData :: Name -> Idris ()
translateData n = do
  i <- getIState
--   jri <- lookupCtxtExact n (idris_records i)
--   maybe "" ( ) jri
  let jti = lookupCtxtExact n (idris_datatypes i)
  (runIO . putStrLn) $ maybe "" (show . con_names) jti

translateNameSpace :: [String] -> Idris ()
translateNameSpace ns = do
    (runIO . putStrLn) ("Namespace is: " ++ show ns)
    names <- namesInNS ns
    (runIO . putStrLn . show) names
    ist <- getIState
    iRenderResult $
        indent
            2
            (vsep
                 (map
                      (\n ->
                            prettyName True False [] n <+>
                            colon <+> (group . align $ pprintDelabTy ist n))
                      names))
    (runIO . putStrLn . show)
        (map
             (\n ->
                   (n, lookupTyName n (tt_ctxt ist)))
             names)
    --      (runIO . putStrLn . show) (map (\n -> (n, lookupTy n (tt_ctxt ist))) names)
    --      (runIO . putStrLn . show)
    --        (map (\n -> (n, lookupCtxt n ((definitions . tt_ctxt) ist))) names)
    mapM_
        translateNamedObject
        names

translateNamedObject :: Name -> Idris ()
translateNamedObject name = do
    ist <- getIState
    (runIO . putStrLn . show) name
    mapM_
        (translateTTDecl name)
        (lookupCtxt name ((definitions . tt_ctxt) ist))

-- defined in Core/Evaluate.hs
-- type TTDecl = (Def, RigCount, Injectivity, Accessibility, Totality, MetaInformation)
-- Hidden => Programs can't access the name at all
-- Public => Programs can access the name and use at will
-- Frozen => Programs can access the name, which doesn't reduce
-- Private => Programs can't access the name, doesn't reduce internally
-- data Accessibility = Hidden | Public | Frozen | Private
translateTTDecl
    :: Name -> TTDecl -> Idris ()
translateTTDecl _ (def,_,_,accessibility,_,_) = translateDef def

translateDef :: Def -> Idris ()
translateDef (Function ty tm) =
    (runIO . putStrLn) $ "Function: " ++ show (ty, tm)
translateDef (TyDecl nt ty) = do
  (runIO . putStrLn) $ "TyDecl: " ++ show nt ++ " " ++ show ty
  translateTerm ty
translateDef (Operator ty _ _) = (runIO . putStrLn) $ "Operator: " ++ show ty
translateDef (CaseOp _ returnType argumentTypes originalDefinition simplifiedDefinition caseDefs) =
    let (ns,sc) = cases_runtime caseDefs
    in do (runIO . putStrLn) $
              "Case: returnType=" ++
              show returnType ++
              "\n" ++
              " argumentTypes=" ++
              show argumentTypes ++
              "\n" ++
              " originalDefinition=" ++
              show originalDefinition ++
              "\n" ++
              " simplifiedDefinition=" ++
              show simplifiedDefinition ++
              "\n" ++ "names=" ++ show ns ++ " SC=" ++ show sc ++ "\n\n"
          (runIO . putStrLn) "Translating SC"
          translateSC sc

translateSC :: SC -> Idris ()
translateSC (Case _ name alts) = do
    (runIO . putStrLn) $ "case " ++ show name ++ " of\n"
    mapM_ translateCaseAlt alts
translateSC (ProjCase tm alts) = do
    (runIO . putStrLn) $ "case " ++ show tm ++ " of\n"
    mapM_ translateCaseAlt alts
translateSC (STerm tm) = do
    (runIO . putStrLn) "STerm"
    translateTerm tm
translateSC (UnmatchedCase s) = (runIO . putStrLn) $ "error " ++ show s
translateSC ImpossibleCase = (runIO . putStrLn) $ "impossible"

translateCaseAlt :: CaseAlt -> Idris ()
translateCaseAlt (ConCase n _ args sc) = do
    (runIO . putStrLn) $ show n ++ "(" ++ showSep (", ") (map show args)
    translateSC sc
translateCaseAlt (FnCase n args sc) = do
    (runIO . putStrLn) $
        "FN " ++ show n ++ "(" ++ showSep (", ") (map show args)
    translateSC sc
translateCaseAlt (ConstCase t sc) = do
    (runIO . putStrLn) $ "ConstCase " ++ show t
    translateSC sc
translateCaseAlt (SucCase n sc) = do
    (runIO . putStrLn) $ "SucCase " ++ show n
    translateSC sc
translateCaseAlt (DefaultCase sc) = do
    (runIO . putStrLn) "DefaultCase "
    translateSC sc

translateTerm :: Term -> Idris ()
translateTerm (P nt n t) = do
    (runIO . putStrLn) $
        "P: " ++ show nt ++ ", name: " ++ show n ++ ", term: " ++ show t
    --   this is causing a loop
    --   translateNamedObject n
    translateTerm
        t -- this is erased
translateTerm (V i) = (runIO . putStrLn) $ "V " ++ show i
translateTerm (Bind n b t) = do
    (runIO . putStrLn) $ "Bind " ++ show n ++ " " ++ show b ++ " " ++ show t
    translateBinder b
    translateTerm t
translateTerm (App _ f a) = do
    (runIO . putStrLn) $
        "App: functionType: " ++ show f ++ ", argument: " ++ show a
    translateTerm f
    translateTerm a
translateTerm (Proj t i) = do
    (runIO . putStrLn) $ "Proj " ++ show t ++ " " ++ show i
    translateTerm t
translateTerm (Constant c) = (runIO . putStrLn) $ "Constant " ++ show c
translateTerm Erased = (runIO . putStrLn) "Erased"
translateTerm Impossible = (runIO . putStrLn) "Impossible"
translateTerm (Inferred t) = do
    (runIO . putStrLn) "Inferred "
    translateTerm t
translateTerm (TType i) = do
    (runIO . putStrLn) $ "TType " ++ show i
translateTerm (UType u) = do
    (runIO . putStrLn) $ "UType " ++ show u

translateBinder :: Binder Term -> Idris ()
translateBinder b = (runIO . putStrLn) $ show b

outputAST :: CTranslUnit -> Idris ()
outputAST ctu = (runIO . writeFile "output.c" . render . pretty) ctu

removeDuplicates :: Eq a => [a] -> [a]
removeDuplicates = rdHelper []
    where rdHelper seen [] = seen
          rdHelper seen (x:xs)
              | x `elem` seen = rdHelper seen xs
              | otherwise = rdHelper (seen ++ [x]) xs
