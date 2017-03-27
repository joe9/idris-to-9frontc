{-# LANGUAGE CPP #-}
{-|
Module      : CmdOptions
Description : A parser for the CmdOptions for the executable.
License     : BSD3
Maintainer  : The Idris Community.
-}

{-# LANGUAGE Arrows #-}

module CmdOptions
  ( runArgParser
  ) where

import Idris.AbsSyntax (getClient, getIBCSubDir, getPkg, getPkgCheck,
                        getPkgClean, getPkgMkDoc, getPkgREPL, getPkgTest,
                        getPort, opt)
import Idris.AbsSyntaxTree
import IRTS.CodegenCommon

import Control.Monad.Trans (lift)
import Control.Monad.Trans.Except (throwE)
import Control.Monad.Trans.Reader (ask)
import Data.Char
import Data.Maybe
#if MIN_VERSION_optparse_applicative(0,13,0)
import Data.Monoid ((<>))
#endif
import Options.Applicative
import Options.Applicative.Arrows
import Options.Applicative.Types (ReadM(..))

import Text.ParserCombinators.ReadP hiding (many, option)

import Safe (lastMay)
import qualified Text.PrettyPrint.ANSI.Leijen as PP

runArgParser :: IO [Opt]
runArgParser = do opts <- execParser $ info parser
                          (fullDesc
                           <> headerDoc   (Just idrisHeader)
                           <> progDescDoc (Just idrisProgDesc)
                           <> footerDoc   (Just idrisFooter)
                          )
                  return $ preProcOpts opts
               where
                 idrisHeader = PP.hsep [PP.text "Idris To 9front C Translator"]
                 idrisProgDesc = PP.vsep [PP.empty,
                                          PP.text "Translates Idris code to 9front C program"
                                          ]
                 idrisFooter = PP.vsep [PP.text "Footer Details"
                                        PP.empty,
                                        PP.text "More details over Idris can be found online here:",
                                        PP.empty,
                                        PP.indent 4 (PP.text "http://www.idris-lang.org/")]

pureArgParser :: [String] -> [Opt]
pureArgParser args = case getParseResult $ execParserPure (prefs idm) (info parser idm) args of
  Just opts -> opts
  Nothing -> []

parser :: Parser [Opt]
parser = runA $ proc () -> do
  files <- asA (many $ argument (fmap Filename str) (metavar "FILES")) -< ()
  k parseVersion >>> A helper -< files
